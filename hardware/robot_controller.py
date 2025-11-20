# hardware/robot_controller.py
"""
Manages all physical movements, states, and hardware interactions
for the Reachy robot. This class is the sole authority on robot motion,
including standard poses, speech motion, and Bluetooth audio synchronization.
"""
import asyncio
import time
import traceback
import platform
from multiprocessing import Process, Queue

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove, AntennaTwitchMove, create_pout_exit_lunge_sequence, create_pout_exit_gentle_sequence
from camera_worker import CameraWorker
from head_wobbler import HeadWobbler
from mediapipe_face_detector import HeadTracker
from daemon_media_wrapper import DaemonMediaWrapper
from speech_offset import SpeechOffsetPlayer
from speech_analyzer import SpeechAnalyzer
from system.audio_coordinator import AudioCoordinator
from system.audio_manager import AudioManager
from state_tracker import StateTracker
from system.input_manager import InputManager
from bluetooth_buffered_capturer import BluetoothBufferedCapturer

class RobotController:
    def __init__(self, state_tracker: StateTracker, input_manager: InputManager):
        print("[INFO] RobotController: Initializing Reachy hardware components...")
        self.daemon_client = DaemonClient("http://localhost:8100")
        self.media_manager = DaemonMediaWrapper("http://localhost:8100")
        self.head_tracker = HeadTracker(device="cpu")
        self.audio_manager = AudioManager()
        self.state_tracker = state_tracker
        self.input_manager = input_manager

        # Debug window process (macOS requires separate process for cv2.imshow)
        self.debug_frame_queue = None
        self.debug_window_process = None
        if platform.system() == "Darwin":
            from debug_window_process import run_debug_window
            self.debug_frame_queue = Queue(maxsize=2)  # Small queue, drop frames if full
            self.debug_window_process = Process(target=run_debug_window, args=(self.debug_frame_queue,))
            self.debug_window_process.start()
            print("[INFO] Debug window process started (macOS)")

        self.camera_worker = CameraWorker(
            media_manager=self.media_manager,
            head_tracker=self.head_tracker,
            daemon_client=self.daemon_client,
            movement_manager=None,
            debug_window=True,  # Enable debug visualization
            debug_frame_queue=self.debug_frame_queue  # Pass queue to worker
        )
        self.movement_manager = MovementManager(
            current_robot=self.daemon_client,
            camera_worker=self.camera_worker
        )
        self.camera_worker.movement_manager = self.movement_manager

        self.head_wobbler = HeadWobbler(
            set_speech_offsets=self.movement_manager.set_speech_offsets
        )
        self.speech_analyzer = SpeechAnalyzer()
        self.speech_offset_player = SpeechOffsetPlayer(self.movement_manager)
        self.speech_motion_cleanup_task = None
        self.speech_motion_duration = 0

        self.audio_coordinator = AudioCoordinator(
            self.audio_manager, self.speech_analyzer, self.speech_offset_player
        )
        
        self.bluetooth_capturer = BluetoothBufferedCapturer(
            self.audio_coordinator,
            self.movement_manager,
            self.state_tracker,
            self._reset_bluetooth_timeout
        )
        self._bluetooth_timeout_task = None
        
        self.state_tracker.register_callback(self._on_state_change)

        self._in_pout_mode = False
        self._pout_move = None
        self._pout_antenna_twitch = None
        self._sleep_move = None
        self._is_physically_asleep = False

        # Pre-analyzed response caches for instant playback
        self.sleep_response_cache = {}
        self.pout_response_cache = {}
        self.quit_baby_response_cache = {}
        self._preload_response_audio()

    def start_threads(self):
        self.camera_worker.start()
        self.movement_manager.start()

    def stop_threads(self):
        self.movement_manager.stop()
        self.camera_worker.stop()

        # Cleanup debug window process
        if self.debug_window_process is not None:
            print("[INFO] Terminating debug window process...")
            self.debug_window_process.terminate()
            self.debug_window_process.join(timeout=2)
            if self.debug_window_process.is_alive():
                self.debug_window_process.kill()
            print("[INFO] Debug window process terminated")

    def _preload_response_audio(self):
        """Pre-analyze all pre-recorded response audio files at startup"""
        from pathlib import Path

        response_folders = {
            "sleep_responses": self.sleep_response_cache,
            "pout_responses": self.pout_response_cache,
            "quit_being_a_baby": self.quit_baby_response_cache
        }

        base_dir = Path("/home/user/reachy/pi_reachy_deployment/assets/sounds/laura")

        for folder_name, cache in response_folders.items():
            folder_path = base_dir / folder_name

            if not folder_path.exists():
                print(f"[PRELOAD] Folder not found: {folder_name} - skipping")
                continue

            audio_files = list(folder_path.glob("*.mp3"))

            if not audio_files:
                print(f"[PRELOAD] No audio files in {folder_name}/")
                continue

            print(f"[PRELOAD] Analyzing {len(audio_files)} files in {folder_name}/...")

            for audio_file in audio_files:
                try:
                    # Analyze audio for speech motion (empty text is fine for pre-recorded)
                    analysis = self.speech_analyzer.analyze(str(audio_file), "")
                    cache[audio_file.name] = {
                        "path": str(audio_file),
                        "analysis": analysis
                    }
                    print(f"[PRELOAD] ✓ {folder_name}/{audio_file.name}")
                except Exception as e:
                    print(f"[PRELOAD] ✗ Failed {audio_file.name}: {e}")

            print(f"[PRELOAD] {folder_name}: {len(cache)} files ready")

    async def play_prerecorded_response(self, cache_dict, response_type="response"):
        """Play a random pre-recorded response with speech motion"""
        import random

        if not cache_dict:
            print(f"[{response_type.upper()}] No pre-loaded responses available")
            return

        # Pick random response
        response_name = random.choice(list(cache_dict.keys()))
        response_data = cache_dict[response_name]

        print(f"[{response_type.upper()}] Playing: {response_name}")

        # Load the pre-analyzed timeline
        self.speech_offset_player.load_timeline(response_data["analysis"])

        # Start playback and speech motion simultaneously
        audio_start_time = time.time()

        # Create audio playback task (non-blocking)
        audio_task = asyncio.create_task(
            self.audio_manager.play_audio(response_data["path"])
        )

        # Start speech motion immediately
        self.speech_offset_player.play(audio_start_time)

        # Wait for audio to complete
        await audio_task

        # Stop speech motion
        self.speech_offset_player.stop()

        print(f"[{response_type.upper()}] Playback complete")

    def _on_state_change(self, new_state, old_state, mood, text):
        from state_tracker import StateTracker
        print(f"[STATE] {old_state} -> {new_state}")

        if new_state == "sleep" and not self._is_physically_asleep:
            self.enter_sleep_mode()
        elif old_state == "sleep" and new_state != "sleep":
            if self._sleep_move:
                self.exit_sleep_mode()

        config = StateTracker.STATE_CONFIGS.get(new_state, {})
        breathing_config = config.get("breathing", {})

        # Control breathing based on state config
        if new_state not in ["speaking", "sleep", "pout"]:
            if breathing_config.get("enabled", False):
                self.movement_manager.resume_breathing()
            else:
                self.movement_manager.pause_breathing()

        # Control face tracking based on state config
        self.camera_worker.set_head_tracking_enabled(config.get("face_tracking", False))

        # Passive mode: no wake word detection (API-driven via Gradio UI)

    def enter_pout_mode(self, initial_body_yaw_deg: float = 0.0, entry_speed: str = "slow"):
        if self._in_pout_mode: return
        self._in_pout_mode = True
        self.state_tracker.update_state("pout")
        self.camera_worker.set_head_tracking_enabled(False)
        
        try:
            head_joints, antenna_joints = self.daemon_client.get_current_joint_positions()
            current_head_pose = self.daemon_client.get_current_head_pose()
            current_antennas = (antenna_joints[-2], antenna_joints[-1])
            current_body_yaw = head_joints[0]
        except Exception as e:
            from reachy_mini.utils import create_head_pose
            print(f"[POUT] Error getting current pose: {e}")
            current_head_pose = create_head_pose(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, degrees=True, mm=False)
            current_antennas = (0.0, 0.0); current_body_yaw = 0.0
        
        duration = 0.5 if entry_speed == "instant" else 2.0
        self._pout_move = PoutPoseMove(
            interpolation_start_pose=current_head_pose, interpolation_start_antennas=current_antennas,
            interpolation_start_body_yaw=current_body_yaw, interpolation_duration=duration,
            target_body_yaw_deg=initial_body_yaw_deg,
        )
        self.movement_manager.resume_from_external_control()
        self.movement_manager.clear_move_queue()
        self.movement_manager.queue_move(self._pout_move)

    def exit_pout_mode(self, exit_duration: float = 3.0):
        if not self._in_pout_mode: return
        self._in_pout_mode = False
        if self._pout_antenna_twitch: self._pout_antenna_twitch = None

        try:
            head_joints, antenna_joints = self.daemon_client.get_current_joint_positions()
            current_head_pose = self.daemon_client.get_current_head_pose()
            current_antennas = (antenna_joints[-2], antenna_joints[-1]); current_body_yaw = head_joints[0]
        except Exception as e:
            print(f"[POUT] Error getting current pose: {e}")
            current_head_pose = PoutPoseMove.SLEEP_HEAD_POSE; current_antennas = tuple(PoutPoseMove.SLEEP_ANTENNAS); current_body_yaw = 0.0
            
        user_face_yaw = self.camera_worker.get_world_frame_target_yaw() if self.camera_worker.is_face_detected() else None
        exit_sequence = create_pout_exit_gentle_sequence(
            current_head_pose, current_antennas, current_body_yaw, user_face_yaw=user_face_yaw, exit_duration=exit_duration
        )
        self.movement_manager.clear_move_queue(); self.movement_manager.queue_move(exit_sequence)
        
        self._pout_move = None
        self.state_tracker.update_state("idle")
        self.movement_manager.resume_breathing()
        self.camera_worker.set_head_tracking_enabled(True)

    def pout_rotate_to(self, angle_deg: float):
        if not self._in_pout_mode or not self._pout_move: return
        self._pout_move.rotate_to(angle_deg)

    def pout_play_antenna_pattern(self, pattern: str = "frustration_twitch"):
        if not self._in_pout_mode: return
        self._pout_antenna_twitch = AntennaTwitchMove(pattern=pattern, base_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS))
        self.movement_manager.queue_move(self._pout_antenna_twitch)

    def enter_sleep_mode(self):
        if self.speech_motion_cleanup_task and not self.speech_motion_cleanup_task.done():
            self.speech_motion_cleanup_task.cancel()
        
        try:
            self.speech_offset_player.stop()
            self.movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        except Exception as e:
            print(f"[SLEEP] Error stopping speech offset player: {e}")

        self.movement_manager.resume_from_external_control()

        try:
            head_joints, antenna_joints = self.daemon_client.get_current_joint_positions()
            current_head_pose = self.daemon_client.get_current_head_pose()
            current_antennas = (antenna_joints[-2], antenna_joints[-1]); current_body_yaw = head_joints[0]
        except Exception as e:
            from reachy_mini.utils import create_head_pose
            print(f"[SLEEP] Error getting current pose: {e}")
            current_head_pose = create_head_pose(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, degrees=True, mm=False)
            current_antennas = (0.0, 0.0); current_body_yaw = 0.0

        self._sleep_move = PoutPoseMove(
            interpolation_start_pose=current_head_pose, interpolation_start_antennas=current_antennas,
            interpolation_start_body_yaw=current_body_yaw, interpolation_duration=2.0, target_body_yaw_deg=0.0,
            antenna_twitch_pattern=None,
        )
        self.movement_manager.clear_move_queue(); self.movement_manager.queue_move(self._sleep_move)
        
        self._is_physically_asleep = True
        self.state_tracker.update_state("sleep")

    def exit_sleep_mode(self):
        try:
            head_joints, antenna_joints = self.daemon_client.get_current_joint_positions()
            current_head_pose = self.daemon_client.get_current_head_pose()
            current_antennas = (antenna_joints[-2], antenna_joints[-1]); current_body_yaw = head_joints[0]
        except Exception as e:
            print(f"[SLEEP] Error getting current pose: {e}")
            current_head_pose = PoutPoseMove.SLEEP_HEAD_POSE; current_antennas = tuple(PoutPoseMove.SLEEP_ANTENNAS); current_body_yaw = 0.0

        exit_sequence = create_pout_exit_gentle_sequence(
            current_head_pose, current_antennas, current_body_yaw, user_face_yaw=None, exit_duration=3.0
        )
        self.movement_manager.clear_move_queue(); self.movement_manager.queue_move(exit_sequence)
        
        self._sleep_move = None
        self._is_physically_asleep = False
        self.movement_manager.resume_breathing()
        self.camera_worker.set_head_tracking_enabled(True)

    async def enter_bluetooth_ready(self):
        print("[BLUETOOTH] Entering bluetooth_ready state")
        self.input_manager.stop_wake_word_detection()
        self.state_tracker.update_state("bluetooth_ready")
        await self.bluetooth_capturer.start()
        self._bluetooth_timeout_task = asyncio.create_task(self._bluetooth_timeout())
        print("[BLUETOOTH] bluetooth_ready active - buffered capture running")

    async def exit_bluetooth_mode(self):
        print("[BLUETOOTH] Exiting Bluetooth mode")
        if hasattr(self, '_bluetooth_timeout_task') and self._bluetooth_timeout_task:
            self._bluetooth_timeout_task.cancel()
            self._bluetooth_timeout_task = None
        await self.bluetooth_capturer.stop()
        self.movement_manager.resume_breathing()
        self.input_manager.restart_wake_word_detection()
        self.state_tracker.update_state("idle")
        print("[BLUETOOTH] Returned to idle state")

    def _reset_bluetooth_timeout(self):
        if hasattr(self, '_bluetooth_timeout_task') and self._bluetooth_timeout_task:
            self._bluetooth_timeout_task.cancel()
        self._bluetooth_timeout_task = asyncio.create_task(self._bluetooth_timeout())

    async def _bluetooth_timeout(self):
        try:
            await asyncio.sleep(120)
            if self.state_tracker.get_state() in ["bluetooth_ready", "bluetooth_playing"]:
                print("[BLUETOOTH] Timeout reached (2 minutes) - exiting Bluetooth mode")
                await self.exit_bluetooth_mode()
        except asyncio.CancelledError:
            print("[BLUETOOTH] Timeout cancelled")
