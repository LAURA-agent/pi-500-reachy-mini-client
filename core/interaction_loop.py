# core/interaction_loop.py
"""
Manages the main user interaction flow: wake word detection,
speech capture, command processing, and conversation handling.
This contains the logic previously in the 'run_main_loop' method.
"""
import asyncio
import traceback
import base64
import cv2

from display.display_manager import DisplayManager
from system.input_manager import InputManager
from hardware.robot_controller import RobotController
from communication.mcp_session_manager import MCPSessionManager
from system.conversation_manager import ConversationManager
from system.system_command_manager import SystemCommandManager
from speech_capture.speech_processor import SpeechProcessor
from processing.command_handler import CommandHandler

class InteractionLoop:
    def __init__(self, state_tracker: DisplayManager, input_manager: InputManager,
                 robot_controller: RobotController, mcp_session_manager: MCPSessionManager,
                 speech_processor: SpeechProcessor, conversation_manager: ConversationManager,
                 system_command_manager: SystemCommandManager):
        
        self.state_tracker = state_tracker
        self.input_manager = input_manager
        self.robot_controller = robot_controller
        self.mcp_session_manager = mcp_session_manager
        self.speech_processor = speech_processor
        self.conversation_manager = conversation_manager
        self.system_command_manager = system_command_manager

        self.command_handler = CommandHandler(
            self.robot_controller, self.mcp_session_manager, self.system_command_manager
        )
        
        self.session_id = None
        self.offline_mode = False
        self._previous_state = None
        self._wake_from_sleep = False
        self._capture_camera_on_wake = False

    def set_session_id(self, session_id: str):
        self.session_id = session_id
        self.command_handler.set_session_id(session_id)
        
    def set_offline_mode(self, is_offline: bool):
        self.offline_mode = is_offline

    async def run(self):
        print("[INFO] Main interaction loop started.")
        while True:
            try:
                current_state = self.state_tracker.get_state()

                if self._previous_state == 'sleep' and current_state != 'sleep':
                    self._wake_from_sleep = True
                elif self._previous_state != current_state:
                    self._wake_from_sleep = False
                self._previous_state = current_state

                from robot_state_machine import RobotStateMachine
                state_config = RobotStateMachine.STATE_CONFIGS.get(current_state, {})
                if state_config.get("wake_word_detection", False):
                    wake_event_source = await self.input_manager.check_for_wake_events()
                    if not wake_event_source:
                        await asyncio.sleep(0.05)
                        continue
                    
                    await self.handle_wake_event(wake_event_source)
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                print("[INFO] Interaction loop cancelled.")
                raise  # Re-raise to propagate cancellation upward
            except KeyboardInterrupt:
                print("[INFO] KeyboardInterrupt in interaction loop - shutting down.")
                raise
            except Exception as e:
                print(f"[ERROR] Critical error in interaction loop: {e}")
                traceback.print_exc()
                self.state_tracker.update_state("idle")
                await asyncio.sleep(1)

    async def handle_wake_event(self, wake_event_source: str):
        if 'wakeword' in wake_event_source:
            self.input_manager.stop_wake_word_detection()

        if self.state_tracker.get_state() == 'sleep':
            self.robot_controller.exit_sleep_mode()
            await asyncio.sleep(1.0)

        # Check for special wake words. This will also set flags for later use.
        is_handled_immediately = await self.command_handler.handle_wake_command(wake_event_source)
        if is_handled_immediately:
            self.input_manager.restart_wake_word_detection()
            return

        self._capture_camera_on_wake = True
        await self.robot_controller.audio_manager.initialize_input()
        self.state_tracker.update_state('listening')
        self.robot_controller.movement_manager.pause_breathing()

        # Adjust VAD timeout based on wake word context
        is_claude_code_mode = self.command_handler.claude_code_mode_requested
        transcript = await self.speech_processor.capture_speech_with_unified_vad(
            self.state_tracker,
            claude_code_mode=is_claude_code_mode
        )
        self.command_handler.claude_code_mode_requested = False # Reset flag

        if not transcript:
            print("[INFO] No speech detected, returning to idle.")
            self._capture_camera_on_wake = False
            self.robot_controller.movement_manager.resume_breathing()
            self.input_manager.restart_wake_word_detection()
            self.state_tracker.update_state("idle")
            return

        self.input_manager.update_last_interaction()

        # Route to Claude Code if in offline mode or if the specific wake word was used
        if is_claude_code_mode or (self.offline_mode and wake_event_source in ["keyboard_code", "keyboard_laura"]):
            print(f"[INFO] Injecting directly to Claude Code: '{transcript}'")
            self.state_tracker.update_state('thinking')
            await self.command_handler.inject_to_claude_code_with_sounds(transcript)
            self.robot_controller.movement_manager.resume_breathing()
            self.state_tracker.update_state('idle')
            self.input_manager.restart_wake_word_detection()
            return

        # Check for text-based system commands (e.g., "change TTS")
        is_cmd, cmd_type, cmd_arg = self.system_command_manager.detect_system_command(transcript)
        if is_cmd:
            self.state_tracker.update_state('thinking')
            await self.system_command_manager.handle_system_command(
                cmd_type, cmd_arg, self.mcp_session_manager.mcp_session,
                self.conversation_manager.tts_handler, self.robot_controller.audio_coordinator, self.state_tracker
            )
            self.robot_controller.movement_manager.resume_breathing()
            self.state_tracker.update_state('idle')
            self.input_manager.restart_wake_word_detection()
            return

        # If online, proceed with normal server-based conversation
        await self.process_conversation(transcript)

    async def process_conversation(self, transcript: str):
        self.state_tracker.update_state('thinking')
        
        wake_frame = None
        if self._capture_camera_on_wake:
            print("[WAKE] Capturing camera frame...")
            try:
                frame = self.robot_controller.media_manager.get_frame()
                if frame is not None:
                    success, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        wake_frame = base64.b64encode(jpeg_bytes.tobytes()).decode('utf-8')
            except Exception as e:
                print(f"[WAKE] Camera frame capture error: {e}")
            finally:
                self._capture_camera_on_wake = False

        if self._wake_from_sleep:
            self._wake_from_sleep = False

        response = await self.mcp_session_manager.send_to_server(transcript, wake_frame=wake_frame)
        
        if response is not None:
            await self.conversation_manager.process_initial_response(
                response, self.state_tracker, self.mcp_session_manager, self.system_command_manager
            )
        else:
            # Transcript was too short or another filter was applied
            self.state_tracker.update_state("idle")
            self.robot_controller.movement_manager.resume_breathing()
            self.input_manager.restart_wake_word_detection()
