"""Movement system with sequential primary moves and additive secondary moves.

Design overview
- Primary moves (emotions, dances, goto, breathing) are mutually exclusive and run
  sequentially.
- Secondary moves (speech sway, face tracking) are additive offsets applied on top
  of the current primary pose.
- There is a single control point to the robot: `ReachyMini.set_target`.
- The control loop runs near 100 Hz and is phase-aligned via a monotonic clock.
- Idle behaviour starts an infinite `BreathingMove` after a short inactivity delay
  unless listening is active.

Threading model
- A dedicated worker thread owns all real-time state and issues `set_target`
  commands.
- Other threads communicate via a command queue (enqueue moves, mark activity,
  toggle listening).
- Secondary offset producers set pending values guarded by locks; the worker
  snaps them atomically.

Units and frames
- Secondary offsets are interpreted as metres for x/y/z and radians for
  roll/pitch/yaw in the world frame (unless noted by `compose_world_offset`).
- Antennas and `body_yaw` are in radians.
- Head pose composition uses `compose_world_offset(primary_head, secondary_head)`;
  the secondary offset must therefore be expressed in the world frame.

Safety
- Listening freezes antennas, then blends them back on unfreeze.
- Interpolations and blends are used to avoid jumps at all times.
- `set_target` errors are rate-limited in logs.
"""

from __future__ import annotations
import time
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import (
    compose_world_offset,
    linear_pose_interpolation,
    time_trajectory,
    InterpolationTechnique,
)


logger = logging.getLogger(__name__)

# Configuration constants
CONTROL_LOOP_FREQUENCY_HZ = 100.0  # Hz - Target frequency for the movement control loop


def ease_in_out(t: float) -> float:
    """Ease-in-out easing function for smooth animation-style interpolation.

    Provides natural acceleration at the start and deceleration at the end,
    creating organic, cartoon-like motion.

    Args:
        t: Progress from 0.0 to 1.0

    Returns:
        Eased progress value from 0.0 to 1.0

    Math: Uses cosine curve: 0.5 - 0.5 * cos(t * π)
    - At t=0: returns 0 (starts slow)
    - At t=0.5: returns 0.5 (middle speed)
    - At t=1: returns 1 (ends slow)
    """
    return 0.5 - 0.5 * np.cos(t * np.pi)

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]  # (head_pose_4x4, antennas, body_yaw)


class AnchorState(Enum):
    """State machine for body yaw anchoring system."""
    ANCHORED = "anchored"        # Body locked at anchor point
    SYNCING = "syncing"          # Body moving to match head
    STABILIZING = "stabilizing"  # Waiting for head to stabilize


class BreathingMove(Move):  # type: ignore
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
        breathing_mode: str = "sway_roll",
        sway_amplitude: float = 0.008,  # 8mm Y sway
        roll_amplitude: float = np.deg2rad(6.0),  # Increased from 2.25° to 6° for visibility
        breathing_frequency: float = 0.2,
        antenna_sway_amplitude: float = np.deg2rad(12),
        antenna_frequency: float = 0.5,
    ):
        """Initialize breathing move.

        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from
            interpolation_duration: Duration of interpolation to neutral (seconds)
            breathing_mode: "sway_roll" (current) or "vertical_only" (original)
            sway_amplitude: Side-to-side sway amplitude in meters (default 5mm)
            roll_amplitude: Roll tilt amplitude in radians (default 2.25 degrees)
            breathing_frequency: Breathing frequency in Hz (default 0.2 Hz = 5s cycle)
            antenna_sway_amplitude: Antenna sway amplitude in radians (default 12 degrees)
            antenna_frequency: Antenna sway frequency in Hz (default 0.5 Hz)

        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        # Neutral positions for breathing base
        # Always use true neutral position (0, 0, 0.01) for breathing baseline
        # This ensures breathing sway rotates correctly in world coordinates
        self.neutral_head_pose = create_head_pose(
            x=0.0,      # True neutral - no forward/back offset
            y=0.0,      # True neutral - no left/right offset
            z=0.01,     # 1.0cm lift to avoid low position
            roll=0.0,
            pitch=0.0,  # Look straight ahead
            yaw=0.0,
            degrees=True,
            mm=False
        )
        self.neutral_antennas = np.array([0.0, 0.0])

        # Breathing parameters - configurable
        self.breathing_mode = breathing_mode
        self.sway_amplitude = sway_amplitude
        self.roll_amplitude = roll_amplitude
        self.breathing_frequency = breathing_frequency
        self.antenna_sway_amplitude = antenna_sway_amplitude
        self.antenna_frequency = antenna_frequency

        # Runtime amplitude scaling (0.0 to 1.0)
        # Used to scale down breathing during external moves without stopping it
        self.amplitude_scale = 1.0

        # Body anchor yaw for coordinate transformation (radians)
        # Updated each tick to rotate sway into body-local coordinates
        self.body_anchor_yaw = 0.0

        # Suppress breathing motion when strain exceeds threshold
        # Prevents collision when head is turned before body follows
        self.suppress_due_to_strain = False

        # Suppress Y sway during speech to prevent interference with speech motion
        self.suppress_y_sway_during_speech = False

        # Track whether body is anchored (stable) vs moving
        # Only apply sway when anchored to prevent camera motion artifacts
        self.is_anchored = True

        # Suppression recovery tracking for smooth resume
        self._was_suppressed = False
        self._recovery_start_time = None
        self._recovery_duration = 0.5  # 500ms to smoothly resume breathing

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous breathing (never ends naturally)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, self.neutral_head_pose, interpolation_t,
            )

            # Interpolate antennas
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)

        else:
            # Phase 2: Gentle breathing - roll + coupled Y sway
            # Y sway is coupled with roll direction to create natural breathing motion
            # Without Y coupling, Stewart platform compensation creates "shoulder slide" effect
            breathing_time = t - self.interpolation_duration

            # Apply amplitude scaling to all motion components
            # This allows breathing to continue at reduced intensity during external moves
            scale = self.amplitude_scale

            # Breathing motion control:
            # - Roll tilt: Suppressed when strain is high (prevents collision)
            # - Y sway: Only active when anchored (prevents camera motion artifacts)

            # Check if currently suppressed
            is_currently_suppressed = self.suppress_due_to_strain or not self.is_anchored

            # Track recovery from suppression
            recovery_scale = 1.0
            if is_currently_suppressed:
                # Currently suppressed - mark it
                self._was_suppressed = True
                self._recovery_start_time = None
                recovery_scale = 0.0
            elif self._was_suppressed:
                # Was suppressed, now resuming - start recovery interpolation
                if self._recovery_start_time is None:
                    self._recovery_start_time = t

                recovery_elapsed = t - self._recovery_start_time
                if recovery_elapsed < self._recovery_duration:
                    # Smooth interpolation from 0 to 1
                    recovery_scale = recovery_elapsed / self._recovery_duration
                else:
                    # Recovery complete
                    recovery_scale = 1.0
                    self._was_suppressed = False

            # DISABLED: Roll tilt and sway causing jerky face tracking
            # Roll tilt - suppressed when strain exceeds threshold
            roll_tilt = 0.0
            # if self.suppress_due_to_strain:
            #     roll_tilt = 0.0
            # else:
            #     roll_tilt = -self.roll_amplitude * scale * recovery_scale * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)

            # Y sway - only when anchored to prevent camera motion feedback
            # Also disabled during speech to prevent interference
            x_sway = 0.0
            y_sway = 0.0
            if self.is_anchored and not self.suppress_y_sway_during_speech:
                # Body-relative sway (side-to-side) - rotate to world coordinates
                # In body frame: X=0, Y=sway (perpendicular to body orientation)
                # Rotate by body anchor yaw to get world X and Y components
                body_local_sway = self.sway_amplitude * scale * recovery_scale * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)

                # Transform body-local Y sway into world coordinates
                # body_anchor_yaw is already in radians (set by MovementManager)
                x_sway = -body_local_sway * np.sin(self.body_anchor_yaw)  # X component in world frame
                y_sway = body_local_sway * np.cos(self.body_anchor_yaw)   # Y component in world frame
            # else:
            #     x_sway = 0.0
            #     y_sway = 0.0

            # Create a relative offset pose containing only the breathing motion
            head_pose = create_head_pose(
                x=x_sway,              # X component from rotated sway
                y=y_sway,              # Y component from rotated sway
                z=0.0,                 # No vertical breathing motion
                roll=np.rad2deg(roll_tilt),
                pitch=0.0,             # Breathing does not affect pitch
                yaw=0.0,               # Breathing does not affect yaw
                degrees=True,
                mm=False
            )

            # Antenna sway (opposite directions) - scaled by amplitude
            antenna_sway = self.antenna_sway_amplitude * scale * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        # Return None for body_yaw to preserve current position (avoid locking to center)
        return (head_pose, antennas, None)


def combine_full_body(primary_pose: FullBodyPose, secondary_pose: FullBodyPose) -> FullBodyPose:
    """Combine primary and secondary full body poses."""
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose

    # The secondary pose (face tracking) is the base, and the primary pose (breathing) is the offset.
    combined_head = compose_world_offset(secondary_head, primary_head, reorthonormalize=True)

    combined_antennas = (
        primary_antennas[0] + secondary_antennas[0],
        primary_antennas[1] + secondary_antennas[1],
    )

    # Handle None body_yaw (BreathingMove returns None to preserve current position)
    if primary_body_yaw is None and secondary_body_yaw is None:
        combined_body_yaw = 0.0
    elif primary_body_yaw is None:
        combined_body_yaw = secondary_body_yaw
    elif secondary_body_yaw is None:
        combined_body_yaw = primary_body_yaw
    else:
        combined_body_yaw = primary_body_yaw + secondary_body_yaw

    return (combined_head, combined_antennas, combined_body_yaw)


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Create a deep copy of a full body pose tuple."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))


@dataclass
class MovementState:
    """State tracking for the movement system."""
    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0
    speech_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
    speech_antennas: Tuple[float, float] = (0.0, 0.0)  # (left_rad, right_rad)
    face_tracking_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        self.last_activity_time = time.monotonic()


@dataclass
class LoopFrequencyStats:
    """Track rolling loop frequency statistics."""
    mean: float = 0.0
    m2: float = 0.0
    min_freq: float = float("inf")
    count: int = 0
    last_freq: float = 0.0
    potential_freq: float = 0.0

    def reset(self) -> None:
        self.mean = 0.0
        self.m2 = 0.0
        self.min_freq = float("inf")
        self.count = 0


class MovementManager:
    """Coordinate sequential moves, additive offsets, and robot output at 100 Hz."""

    def __init__(
        self,
        current_robot: ReachyMini,
        camera_worker: "Any" = None,
    ):
        self.current_robot = current_robot
        self.camera_worker = camera_worker
        self._now = time.monotonic
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        neutral_pose = create_head_pose(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, degrees=True, mm=False)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)
        self.move_queue: deque[Move] = deque()
        self.idle_inactivity_delay = 0.3
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency
        self._last_breathing_debug = 0.0
        self._breathing_debug_interval = 2.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_listening = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(self.state.last_primary_pose)
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4
        self._last_listening_blend_time = self._now()
        self._breathing_active = False
        self._current_breathing_roll = 0.0
        self._breathing_roll_lock = threading.Lock()
        self._breathing_scale = 1.0
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 1.0
        self._set_target_err_suppressed = 0
        self._suppress_face_tracking = False
        self._external_control_active = False
        self._external_control_lock = threading.Lock()
        self._anchor_state = AnchorState.ANCHORED
        self._body_anchor_yaw = 0.0
        self._current_body_yaw_deg = 0.0
        self._strain_threshold_deg = 13.0
        self._breathing_suppress_threshold_deg = 10.0
        self._current_strain_deg = 0.0
        self._stability_duration_s = 1.5
        self._stability_threshold_deg = 2.0
        self._last_head_yaw_deg = 0.0
        self._head_stable_since = None
        # Antenna alert behavior during body repositioning
        self._antenna_alert_start_time: float | None = None
        self._antenna_alert_resume_delay = 0.5  # Resume breathing after 0.5s, not full completion
        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()
        self._speech_offsets_lock = threading.Lock()
        self._pending_speech_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._pending_speech_antennas: Tuple[float, float] = (0.0, 0.0)
        self._speech_offsets_dirty = False
        self._speech_session_active = False  # Persistent flag for entire speech duration
        self._face_offsets_lock = threading.Lock()
        self._pending_face_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._face_offsets_dirty = False
        # Low-pass filter for face tracking to prevent 100Hz jitter with breathing
        self._smoothed_face_offsets: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._face_tracking_smoothing = 0.92  # 0.92 = smooth, 0.0 = instant (no filter)
        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = self._is_listening
        self._status_lock = threading.Lock()
        self._freq_stats = LoopFrequencyStats()
        self._freq_snapshot = LoopFrequencyStats()

    def queue_move(self, move: Move) -> None:
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        self._command_queue.put(("clear_queue", None))

    def set_speech_offsets(self, offsets: Tuple[float, float, float, float, float, float], antennas: Tuple[float, float] = (0.0, 0.0)) -> None:
        with self._speech_offsets_lock:
            self._pending_speech_offsets = offsets
            self._pending_speech_antennas = antennas
            self._speech_offsets_dirty = True

            # Track speech session: True when any non-zero values, False when all zeros
            # This provides persistent suppression for entire speech duration
            has_nonzero = any(abs(val) > 0.001 for val in offsets) or any(abs(val) > 0.001 for val in antennas)
            self._speech_session_active = has_nonzero

    def get_current_breathing_roll(self) -> float:
        with self._breathing_roll_lock:
            return self._current_breathing_roll

    def get_current_body_yaw_rad(self) -> float:
        """Get current body yaw in radians (for coordinate transformations)."""
        return np.deg2rad(self._current_body_yaw_deg)

    def set_listening(self, listening: bool) -> None:
        with self._shared_state_lock:
            if self._shared_is_listening == listening:
                return
        self._command_queue.put(("set_listening", listening))

    def scale_breathing_down(self) -> None:
        """Signal the control loop to scale breathing down."""
        with self._external_control_lock:
            self._external_control_active = True
        logger.info("Scaling breathing DOWN for external control.")

    def scale_breathing_up(self) -> None:
        """Signal the control loop to scale breathing up."""
        print("[RESUME] scale_breathing_up() called")
        print(f"[RESUME] breathing_scale: {self._breathing_scale:.2f}")
        print(f"[RESUME] current_move: {type(self.state.current_move).__name__ if self.state.current_move else 'None'}")

        if isinstance(self.state.current_move, BreathingMove):
            print(f"[RESUME] BreathingMove.amplitude_scale: {self.state.current_move.amplitude_scale:.2f}")
            print(f"[RESUME] BreathingMove.suppress_due_to_strain: {self.state.current_move.suppress_due_to_strain}")

        print(f"[RESUME] Thread alive: {self._thread.is_alive() if self._thread else False}")

        with self._external_control_lock:
            self._external_control_active = False
            print("[RESUME] Cleared external control flag")

        print("[RESUME] Control loop will resume on next iteration")

    def pause_breathing(self) -> None:
        self.scale_breathing_down()

    def resume_breathing(self) -> None:
        self.scale_breathing_up()

    def _calculate_current_strain(self) -> None:
        if hasattr(self, '_last_commanded_pose'):
            head_pose = self._last_commanded_pose[0]
            head_yaw_rad = self._extract_yaw_from_pose(head_pose)
            head_yaw_deg = np.rad2deg(head_yaw_rad)
            strain = head_yaw_deg - self._body_anchor_yaw
            self._current_strain_deg = (strain + 180) % 360 - 180
        else:
            self._current_strain_deg = 0.0

    def _update_breathing_scale(self) -> None:
        """Update the breathing scale based on the external control flag."""
        with self._external_control_lock:
            external_active = self._external_control_active

        old_scale = self._breathing_scale

        if external_active and self._breathing_scale > 0.0:
            self._breathing_scale = max(0.0, self._breathing_scale - 0.02)  # Ramp down in 0.5s
        elif not external_active and self._breathing_scale < 1.0:
            self._breathing_scale = min(1.0, self._breathing_scale + 0.01)  # Ramp up in 1.0s

        # Debug: Log scale changes (first 20 changes after resume)
        if abs(self._breathing_scale - old_scale) > 0.01:
            if not hasattr(self, '_scale_change_count'):
                self._scale_change_count = 0
            if self._scale_change_count < 20:
                self._scale_change_count += 1
                print(f"[SCALE] {old_scale:.2f} -> {self._breathing_scale:.2f} (external: {external_active})")

        if isinstance(self.state.current_move, BreathingMove):
            self.state.current_move.amplitude_scale = self._breathing_scale
            self.state.current_move.body_anchor_yaw = np.deg2rad(self._body_anchor_yaw)
            self.state.current_move.suppress_due_to_strain = abs(self._current_strain_deg) > self._breathing_suppress_threshold_deg
            self.state.current_move.is_anchored = (self._anchor_state == AnchorState.ANCHORED)

    def _poll_signals(self, current_time: float) -> None:
        self._apply_pending_offsets()
        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)

    def _apply_pending_offsets(self) -> None:
        if self._speech_offsets_dirty:
            with self._speech_offsets_lock:
                self.state.speech_offsets = self._pending_speech_offsets
                self.state.speech_antennas = self._pending_speech_antennas
                self._speech_offsets_dirty = False

                # Suppress Y sway during speech session (persistent flag)
                # Uses _speech_session_active which stays True for entire speech duration
                if isinstance(self.state.current_move, BreathingMove):
                    if self.state.current_move.suppress_y_sway_during_speech != self._speech_session_active:
                        print(f"[Y Sway] Suppression: {self._speech_session_active}")
                    self.state.current_move.suppress_y_sway_during_speech = self._speech_session_active
        if self._face_offsets_dirty:
            with self._face_offsets_lock:
                self.state.face_tracking_offsets = self._pending_face_offsets
                self._face_offsets_dirty = False

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        if command == "queue_move":
            move_name = payload.__class__.__name__
            move_duration = getattr(payload, 'duration', 'N/A')
            print(f"[MOVEMENT] Queuing move: {move_name}, duration: {move_duration}s")
            self.move_queue.append(payload)
            self.state.update_activity()
        elif command == "clear_queue":
            print(f"[MOVEMENT] Clearing move queue ({len(self.move_queue)} moves)")
            self.move_queue.clear()
            self.state.current_move = None
            self._breathing_active = False
            self._suppress_face_tracking = False
        elif command == "set_listening":
            self._is_listening = bool(payload)
            if self._is_listening:
                self._listening_antennas = self._last_commanded_pose[1]
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()

    def _publish_shared_state(self) -> None:
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    def _manage_move_queue(self, current_time: float) -> None:
        move_ended = self.state.current_move and \
                     (current_time - self.state.move_start_time >= self.state.current_move.duration)
        if move_ended:
            move_name = self.state.current_move.__class__.__name__
            duration = self.state.current_move.duration
            print(f"[MOVEMENT] Move completed: {move_name} (duration: {duration}s)")
        if not self.state.current_move or move_ended:
            self.state.current_move = None
            self._suppress_face_tracking = False
            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)
                self._suppress_face_tracking = not self._breathing_active
                move_name = self.state.current_move.__class__.__name__
                duration = self.state.current_move.duration
                print(f"[MOVEMENT] Starting move: {move_name} (duration: {duration}s)")


    def _manage_breathing(self, current_time: float) -> None:
        with self._external_control_lock:
            external_control = self._external_control_active
        idle_for = current_time - self.state.last_activity_time
        
        can_start_breathing = (not self.state.current_move and not self.move_queue and
                               not self._is_listening and not self._breathing_active and
                               not external_control and idle_for >= self.idle_inactivity_delay)
        if can_start_breathing:
            self._breathing_active = True
            try:
                _, current_joints = self.current_robot.get_current_joint_positions()
                current_head_pose = self.current_robot.get_current_head_pose()
                current_antennas = (current_joints[-2], current_joints[-1])
                self.move_queue.append(BreathingMove(current_head_pose, current_antennas, 3.0))
                logger.info("Started breathing after %.1fs of inactivity", idle_for)
            except Exception as e:
                self._breathing_active = False
                logger.error("Failed to start breathing: %s", e)

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        if self.state.current_move:
            move_time = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(move_time)
            if head is None: head = self.state.last_primary_pose[0]
            if antennas is None: antennas = self.state.last_primary_pose[1]
            if body_yaw is None: body_yaw = self.state.last_primary_pose[2]

            # Update breathing roll for camera zero-crossing detection
            if isinstance(self.state.current_move, BreathingMove):
                # Extract roll from the head pose matrix
                from scipy.spatial.transform import Rotation as R
                rotation = R.from_matrix(head[:3, :3])
                roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
                with self._breathing_roll_lock:
                    self._current_breathing_roll = roll

            primary_pose = (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))
            self.state.last_primary_pose = clone_full_body_pose(primary_pose)
            return primary_pose
        return clone_full_body_pose(self.state.last_primary_pose)

    def _extract_yaw_from_pose(self, head_pose: NDArray[np.float32]) -> float:
        return np.arctan2(head_pose[1, 0], head_pose[0, 0])

    def _apply_body_follow(self, pose: FullBodyPose) -> FullBodyPose:
        head_pose, antennas, original_body_yaw = pose

        # CRITICAL: If current move is PoutPoseMove, pass through its body_yaw unchanged
        # Pout mode requires explicit body control without body-follow interference
        if isinstance(self.state.current_move, PoutPoseMove):
            return (head_pose, antennas, original_body_yaw)

        # Use world-frame target from camera_worker for strain calculation if available
        # This solves the "shrinking target" problem where body-relative offsets
        # make it impossible for body-follow to see the true target
        if self.camera_worker is not None:
            world_target_yaw_rad = self.camera_worker.get_world_frame_target_yaw()
            if world_target_yaw_rad is not None:
                # Use world-frame target for strain calculation (Solution #1)
                head_yaw_deg = np.rad2deg(world_target_yaw_rad)
            else:
                # Fall back to composed head pose (no active face tracking)
                head_yaw_deg = np.rad2deg(self._extract_yaw_from_pose(head_pose))
        else:
            # No camera worker, use composed head pose
            head_yaw_deg = np.rad2deg(self._extract_yaw_from_pose(head_pose))

        now = self._now()
        strain = (head_yaw_deg - self._body_anchor_yaw + 180) % 360 - 180
        self._current_strain_deg = strain

        # Track previous state to detect transitions
        prev_state = self._anchor_state

        if self._anchor_state == AnchorState.ANCHORED:
            if abs(strain) > self._strain_threshold_deg:
                # Entering SYNCING - trigger antenna alert
                self._anchor_state = AnchorState.SYNCING
                self._antenna_alert_start_time = now  # Start alert timer
                self._current_body_yaw_deg = self._body_anchor_yaw
            body_yaw = np.deg2rad(self._body_anchor_yaw)
        elif self._anchor_state == AnchorState.SYNCING:
            change = (head_yaw_deg - self._current_body_yaw_deg + 180) % 360 - 180
            self._current_body_yaw_deg += np.clip(change, -0.45, 0.45)
            if abs(head_yaw_deg - self._current_body_yaw_deg) < 3.0:
                self._anchor_state = AnchorState.STABILIZING
                self._head_stable_since = None
                # Keep alert timer running through STABILIZING
            body_yaw = np.deg2rad(self._current_body_yaw_deg)
        elif self._anchor_state == AnchorState.STABILIZING:
            self._current_body_yaw_deg = head_yaw_deg
            if abs(head_yaw_deg - self._last_head_yaw_deg) > self._stability_threshold_deg:
                # Head moved again - if significant, might re-enter SYNCING
                # Check if we need to restart alert
                if prev_state == AnchorState.STABILIZING:
                    # Still in STABILIZING but head moved, reset timer for potential re-sync
                    pass
                self._head_stable_since = None
            elif self._head_stable_since is None:
                self._head_stable_since = now
            elif now - self._head_stable_since >= self._stability_duration_s:
                self._body_anchor_yaw = head_yaw_deg
                self._anchor_state = AnchorState.ANCHORED
                self._antenna_alert_start_time = None  # Clear alert timer
            body_yaw = np.deg2rad(self._current_body_yaw_deg)

        # Detect transition back to SYNCING from STABILIZING (head moved significantly)
        if prev_state == AnchorState.STABILIZING and self._anchor_state == AnchorState.SYNCING:
            # Re-entering SYNCING - restart antenna alert
            self._antenna_alert_start_time = now

        # Antenna alert: Snap to 0° initially, resume breathing after 0.5s
        # If we re-enter SYNCING (head moves again), snap back to 0
        antenna_alert_active = False
        if self._antenna_alert_start_time is not None:
            elapsed = now - self._antenna_alert_start_time
            if elapsed < self._antenna_alert_resume_delay:
                # Still in alert phase - antennas at 0
                antenna_alert_active = True

        if antenna_alert_active:
            antennas = (0.0, 0.0)

        self._last_head_yaw_deg = head_yaw_deg

        # NOTE: No rotation compensation needed for normal face tracking
        # Face tracking offsets are already in world frame and compose_world_offset handles them correctly
        # PoutPoseMove handles its own body-relative head rotation internally (lines 909-918, 944-956)

        return (head_pose, antennas, body_yaw)

    def _compose_full_body_pose(self, current_time: float) -> FullBodyPose:
        primary = self._get_primary_pose(current_time)
        secondary = self._get_secondary_pose()
        composed = combine_full_body(primary, secondary)
        return self._apply_body_follow(composed)

    def _get_secondary_pose(self) -> FullBodyPose:
        # Face tracking offsets: already in world frame from IK (uses current head pose as base)
        # Speech offsets: apply as relative motion on top of face tracking position

        # Create face tracking pose first (base position)
        face_pose = create_head_pose(*self.state.face_tracking_offsets, degrees=False, mm=False)

        # Create speech offset pose (relative motion)
        speech_pose = create_head_pose(*self.state.speech_offsets, degrees=False, mm=False)

        # Compose: apply speech motion relative to face tracking position
        # This makes speech nods happen from wherever face tracking has positioned the head
        head_pose = compose_world_offset(face_pose, speech_pose, reorthonormalize=True)

        # Return speech antennas (face tracking doesn't control antennas)
        return (head_pose, self.state.speech_antennas, 0.0)

    def _update_primary_motion(self, current_time: float) -> None:
        self._manage_move_queue(current_time)
        self._manage_breathing(current_time)

    def _calculate_blended_antennas(self, target_antennas: Tuple[float, float]) -> Tuple[float, float]:
        now = self._now()
        dt = now - self._last_listening_blend_time
        self._last_listening_blend_time = now
        if self._is_listening:
            self._antenna_unfreeze_blend = 0.0
            return self._listening_antennas
        
        blend = min(1.0, self._antenna_unfreeze_blend + dt / self._antenna_blend_duration)
        self._antenna_unfreeze_blend = blend
        return (
            self._listening_antennas[0] * (1.0 - blend) + target_antennas[0] * blend,
            self._listening_antennas[1] * (1.0 - blend) + target_antennas[1] * blend,
        )

    def _issue_control_command(self, head: NDArray[np.float32], antennas: Tuple[float, float], body_yaw: float) -> None:
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))
            # Reset failure counter on success
            if hasattr(self, '_set_target_failures'):
                self._set_target_failures = 0
        except Exception as e:
            now = self._now()

            # Track consecutive failures
            if not hasattr(self, '_set_target_failures'):
                self._set_target_failures = 0
            self._set_target_failures += 1

            # Circuit breaker: if 50 consecutive failures (0.5s at 100Hz), stop the loop
            if self._set_target_failures >= 50:
                print(f"[CRITICAL] 50 consecutive set_target failures - STOPPING CONTROL LOOP")
                print(f"[CRITICAL] Last error: {e}")
                self._stop_event.set()
                return

            if now - self._last_set_target_err >= self._set_target_err_interval:
                print(f"[ERROR] Failed to set robot target ({self._set_target_failures} consecutive): {e}")
                self._last_set_target_err = now
                self._set_target_err_suppressed = 0
            else:
                self._set_target_err_suppressed += 1

    def _update_face_tracking(self, current_time: float) -> None:
        with self._external_control_lock:
            external_control = self._external_control_active
        is_suppressed = self._suppress_face_tracking or external_control or not isinstance(self.state.current_move, BreathingMove)

        if is_suppressed or self.camera_worker is None:
            raw_offsets = (0.0,) * 6
        else:
            raw_offsets = self.camera_worker.get_face_tracking_offsets()

        # Apply low-pass filter to prevent 100Hz jitter fighting with breathing
        # EMA: smoothed = α * smoothed_prev + (1 - α) * raw
        # Higher α (0.9-0.95) = more smoothing, slower response
        alpha = self._face_tracking_smoothing
        for i in range(6):
            self._smoothed_face_offsets[i] = (
                alpha * self._smoothed_face_offsets[i] +
                (1.0 - alpha) * raw_offsets[i]
            )

        self.state.face_tracking_offsets = tuple(self._smoothed_face_offsets)

    def start(self) -> None:
        if self._thread and self._thread.is_alive(): return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def working_loop(self) -> None:
        """Main 100Hz control loop."""
        logger.info("Starting enhanced movement control loop (100Hz)")
        while not self._stop_event.is_set():
            loop_start = self._now()

            # The loop always runs. The _external_control_active flag now only
            # controls the *intensity* of the breathing motion via _breathing_scale.
            self._poll_signals(loop_start)
            self._calculate_current_strain()
            self._update_breathing_scale()

            with self._external_control_lock:
                is_externally_controlled = self._external_control_active

            # When external control is active, STOP sending commands entirely
            # External controller (mood system) has full control of the daemon
            if not is_externally_controlled:
                # Log first few iterations after resuming
                if not hasattr(self, '_resume_iteration_count'):
                    self._resume_iteration_count = 0
                if self._resume_iteration_count < 10:
                    self._resume_iteration_count += 1
                    if self._resume_iteration_count == 1:
                        print(f"[LOOP] Control resumed - breathing_scale={self._breathing_scale:.2f}")

                self._update_primary_motion(loop_start)
                self._update_face_tracking(loop_start)

                # Compose final pose from primary and secondary sources
                head, antennas, body_yaw = self._compose_full_body_pose(loop_start)
                antennas_cmd = self._calculate_blended_antennas(antennas)

                # Send control command to daemon
                self._issue_control_command(head, antennas_cmd, body_yaw)
            else:
                # Reset resume counter when externally controlled
                self._resume_iteration_count = 0

            self._publish_shared_state()
            
            sleep_time = self.target_period - (self._now() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")


# ============================================================================
# Pout Mode Movement Classes
# ============================================================================

class PoutPoseMove(Move):  # type: ignore
    """Maintains sleep/pout pose with optional discrete body rotations.

    Head stays in SLEEP_HEAD_POSE (hunched/hiding posture).
    Breathing is frozen (no sway, no roll).
    Body can rotate to discrete angles: 0°, ±45°, ±90°.
    """

    # SLEEP_HEAD_POSE from reachy_mini SDK (reachy_mini.py:43-50)
    SLEEP_HEAD_POSE = np.array([
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    SLEEP_ANTENNAS = np.array([-3.05, 3.05], dtype=np.float64)

    # Discrete rotation angles (degrees)
    ALLOWED_ROTATIONS = [-90, -45, -30, 0, 30, 45, 90]

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_start_body_yaw: float,
        interpolation_duration: float = 2.0,
        target_body_yaw_deg: float = 0.0,
        antenna_twitch_pattern: str | None = None,
    ):
        """Initialize pout pose move.

        Args:
            interpolation_start_pose: Current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions
            interpolation_start_body_yaw: Current body yaw (radians)
            interpolation_duration: Duration to reach pout pose (seconds)
            target_body_yaw_deg: Target body rotation in degrees (0, ±45, ±90)
        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_start_body_yaw = interpolation_start_body_yaw
        self.interpolation_duration = interpolation_duration

        # Validate and set target rotation
        if target_body_yaw_deg not in self.ALLOWED_ROTATIONS:
            logger.warning(f"Invalid rotation {target_body_yaw_deg}°, using closest allowed value")
            target_body_yaw_deg = min(self.ALLOWED_ROTATIONS, key=lambda x: abs(x - target_body_yaw_deg))

        self.target_body_yaw_rad = np.deg2rad(target_body_yaw_deg)
        self._reached_pout = False

        # Antenna twitch support
        self.antenna_twitch_pattern = antenna_twitch_pattern
        if antenna_twitch_pattern is not None:
            self._twitch_enabled = True
            self._last_flick_time = 0.0
            self._next_flick_interval = self._random_flick_interval()
            self._in_flick = False
            self._flick_duration = 0.1  # 100ms per flick
            self._flick_start_time = 0.0

            # Pattern-specific targets
            if antenna_twitch_pattern == "frustration_twitch":
                self._flick_target_left = -2.5
                self._flick_target_right = 2.5
            elif antenna_twitch_pattern == "angry_swing":
                self._flick_target_left = -2.0
                self._flick_target_right = 2.0
            elif antenna_twitch_pattern == "nervous_flutter":
                self._flick_target_left = -2.8
                self._flick_target_right = 2.8
            else:
                self._flick_target_left = -2.5
                self._flick_target_right = 2.5
        else:
            self._twitch_enabled = False

    def _random_flick_interval(self) -> float:
        """Generate random interval between flicks (1.0-1.5s)."""
        import random
        return random.uniform(1.0, 1.5)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous pout (until explicitly changed)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate pout move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to pout pose
            interp_t = t / self.interpolation_duration

            # Interpolate head to SLEEP_HEAD_POSE (without rotation)
            base_head_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                self.SLEEP_HEAD_POSE,
                interp_t,
            )

            # Interpolate antennas to SLEEP_ANTENNAS
            antennas = (
                (1 - interp_t) * self.interpolation_start_antennas +
                interp_t * self.SLEEP_ANTENNAS
            ).astype(np.float64)

            # Interpolate body yaw
            body_yaw = (
                (1 - interp_t) * self.interpolation_start_body_yaw +
                interp_t * self.target_body_yaw_rad
            )

            # Rotate head pose by body_yaw so head rotates WITH body
            cos_yaw = np.cos(body_yaw)
            sin_yaw = np.sin(body_yaw)
            yaw_rotation = np.array([
                [cos_yaw, -sin_yaw, 0, 0],
                [sin_yaw,  cos_yaw, 0, 0],
                [0,        0,       1, 0],
                [0,        0,       0, 1]
            ], dtype=np.float32)
            head_pose = yaw_rotation @ base_head_pose

        else:
            # Phase 2: Hold pout pose (frozen breathing)
            if not self._reached_pout:
                self._reached_pout = True
                logger.info("Pout pose reached - frozen breathing")

            antennas = self.SLEEP_ANTENNAS.copy()
            body_yaw = self.target_body_yaw_rad

            # Handle smooth rotation if active
            if hasattr(self, '_smooth_rotation_active') and self._smooth_rotation_active:
                import time
                elapsed = time.monotonic() - self._smooth_start_time
                if elapsed < self._smooth_duration:
                    # Still interpolating
                    progress = elapsed / self._smooth_duration
                    body_yaw = self._smooth_start_yaw + (self._smooth_target_yaw - self._smooth_start_yaw) * progress
                else:
                    # Finished interpolating
                    body_yaw = self._smooth_target_yaw
                    self.target_body_yaw_rad = self._smooth_target_yaw  # Update actual target
                    self._smooth_rotation_active = False
                    logger.info(f"Smooth rotation complete at {np.rad2deg(body_yaw):.1f}°")

            # CRITICAL: Rotate head pose by body_yaw so head rotates WITH body
            # Create rotation matrix around Z-axis (yaw)
            cos_yaw = np.cos(body_yaw)
            sin_yaw = np.sin(body_yaw)
            yaw_rotation = np.array([
                [cos_yaw, -sin_yaw, 0, 0],
                [sin_yaw,  cos_yaw, 0, 0],
                [0,        0,       1, 0],
                [0,        0,       0, 1]
            ], dtype=np.float32)

            # Apply rotation: rotated_pose = yaw_rotation @ base_pose
            head_pose = yaw_rotation @ self.SLEEP_HEAD_POSE

        # Apply antenna twitch if enabled
        if self._twitch_enabled:
            antennas = self._apply_antenna_twitch(t, antennas)

        return (head_pose, antennas, body_yaw)

    def _apply_antenna_twitch(self, t: float, base_antennas: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply antenna twitch pattern on top of base position."""
        # Check if it's time for a new flick
        if not self._in_flick and (t - self._last_flick_time) >= self._next_flick_interval:
            self._in_flick = True
            self._flick_start_time = t
            self._last_flick_time = t
            self._next_flick_interval = self._random_flick_interval()

        # Execute flick if active
        if self._in_flick:
            flick_elapsed = t - self._flick_start_time
            if flick_elapsed < self._flick_duration:
                # Quick flick motion (sine wave for smooth motion)
                flick_progress = flick_elapsed / self._flick_duration
                blend = np.sin(np.pi * flick_progress)  # 0 -> 1 -> 0

                # Interpolate from base position to flick target
                left_antenna = base_antennas[0] + (self._flick_target_left - base_antennas[0]) * blend
                right_antenna = base_antennas[1] + (self._flick_target_right - base_antennas[1]) * blend

                return np.array([left_antenna, right_antenna], dtype=np.float64)
            else:
                # Flick complete
                self._in_flick = False

        return base_antennas

    def rotate_to(self, angle_deg: float) -> None:
        """Update target body rotation angle instantly.

        Args:
            angle_deg: Target rotation in degrees (0, ±45, ±90)
        """
        if angle_deg not in self.ALLOWED_ROTATIONS:
            logger.warning(f"Invalid rotation {angle_deg}°, ignoring")
            return

        self.target_body_yaw_rad = np.deg2rad(angle_deg)
        logger.info(f"Pout body rotation target updated to {angle_deg}°")

    def rotate_to_smooth(self, angle_deg: float, duration: float = 3.0) -> None:
        """Smoothly rotate to target angle over specified duration.

        Args:
            angle_deg: Target rotation in degrees (0, ±45, ±90)
            duration: Time to complete rotation in seconds (default: 3.0)
        """
        if angle_deg not in self.ALLOWED_ROTATIONS:
            logger.warning(f"Invalid rotation {angle_deg}°, ignoring")
            return

        import time
        # Store rotation transition state
        if not hasattr(self, '_smooth_rotation_active'):
            self._smooth_rotation_active = False
            self._smooth_start_time = None
            self._smooth_start_yaw = self.target_body_yaw_rad
            self._smooth_target_yaw = self.target_body_yaw_rad
            self._smooth_duration = duration

        # Start new smooth rotation
        self._smooth_rotation_active = True
        self._smooth_start_time = time.monotonic()
        self._smooth_start_yaw = self.target_body_yaw_rad
        self._smooth_target_yaw = np.deg2rad(angle_deg)
        self._smooth_duration = duration

        logger.info(f"Starting smooth rotation from {np.rad2deg(self._smooth_start_yaw):.1f}° to {angle_deg}° over {duration}s")


class AntennaTwitchMove(Move):  # type: ignore
    """Antenna twitch patterns for emotional expression.

    Provides predefined twitch patterns like frustration_twitch (rapid flicks).
    Can layer on top of other moves (antennas only, no head movement).
    """

    def __init__(
        self,
        pattern: str = "frustration_twitch",
        base_antennas: Tuple[float, float] = (0.0, 0.0),
    ):
        """Initialize antenna twitch move.

        Args:
            pattern: Pattern name ("frustration_twitch", "angry_swing", "nervous_flutter")
            base_antennas: Base antenna position to return to between twitches
        """
        self.pattern = pattern
        self.base_antennas = np.array(base_antennas)
        self._last_flick_time = 0.0
        self._next_flick_interval = self._random_flick_interval()
        self._in_flick = False
        self._flick_duration = 0.1  # 100ms per flick
        self._flick_start_time = 0.0

        # Pattern-specific parameters
        # Note: Antenna range is -3 to 0 (left) and 0 to +3 (right)
        # where -3/+3 = fully down, 0 = straight up
        if pattern == "frustration_twitch":
            # Flick from down (-3.05/+3.05) to partially raised (-2.5/+2.5)
            self._flick_target_left = -2.5
            self._flick_target_right = 2.5
        elif pattern == "angry_swing":
            # Larger swing, more raised (-2.0/+2.0)
            self._flick_target_left = -2.0
            self._flick_target_right = 2.0
        elif pattern == "nervous_flutter":
            # Small flutter, barely raised (-2.8/+2.8)
            self._flick_target_left = -2.8
            self._flick_target_right = 2.8
        else:
            logger.warning(f"Unknown pattern '{pattern}', using frustration_twitch")
            self._flick_target_left = -2.5
            self._flick_target_right = 2.5

    def _random_flick_interval(self) -> float:
        """Generate random interval between flicks (1.0-1.5s)."""
        import random
        return random.uniform(1.0, 1.5)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous twitch pattern

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate antenna twitch at time t."""
        # Check if it's time for a new flick
        if not self._in_flick and (t - self._last_flick_time) >= self._next_flick_interval:
            self._in_flick = True
            self._flick_start_time = t
            self._last_flick_time = t
            self._next_flick_interval = self._random_flick_interval()

        # Execute flick if active
        if self._in_flick:
            flick_elapsed = t - self._flick_start_time
            if flick_elapsed < self._flick_duration:
                # Quick flick motion (sine wave for smooth motion)
                # Goes from base -> target -> base over flick_duration
                flick_progress = flick_elapsed / self._flick_duration
                blend = np.sin(np.pi * flick_progress)  # 0 -> 1 -> 0

                # Interpolate from base position to flick target
                left_antenna = self.base_antennas[0] + (self._flick_target_left - self.base_antennas[0]) * blend
                right_antenna = self.base_antennas[1] + (self._flick_target_right - self.base_antennas[1]) * blend

                antennas = np.array([left_antenna, right_antenna], dtype=np.float64)
            else:
                # Flick complete, return to base
                self._in_flick = False
                antennas = self.base_antennas.copy()
        else:
            # Between flicks, hold base position
            antennas = self.base_antennas.copy()

        # No head movement, no body rotation (antennas only)
        return (None, antennas, None)


class SequenceMove(Move):  # type: ignore
    """Base class for choreographed multi-step movement sequences.

    Sequences are non-interruptible and guarantee completion.
    Each step is defined by a target pose, duration, and optional callback.
    """

    def __init__(
        self,
        current_head_pose: NDArray[np.float32],
        current_antennas: Tuple[float, float],
        current_body_yaw: float,
    ):
        """Initialize sequence move.

        Args:
            current_head_pose: Starting head pose
            current_antennas: Starting antenna positions
            current_body_yaw: Starting body yaw (radians)
        """
        # Store initial pose for interpolation from current position
        self.initial_head_pose = current_head_pose.copy()
        self.initial_antennas = np.array(current_antennas)
        self.initial_body_yaw = current_body_yaw

        # Track current values as steps are added
        self.current_head_pose = current_head_pose
        self.current_antennas = np.array(current_antennas)
        self.current_body_yaw = current_body_yaw

        # Sequence steps: list of (duration, target_pose, target_antennas, target_body_yaw)
        self.steps: list[tuple[float, NDArray[np.float32], np.ndarray, float]] = []
        self._total_duration = 0.0
        self._step_start_times: list[float] = []
        self._current_step_index = 0
        self._sequence_complete = False

    def add_step(
        self,
        duration: float,
        target_head_pose: NDArray[np.float32] | None = None,
        target_antennas: Tuple[float, float] | None = None,
        target_body_yaw: float | None = None,
    ) -> None:
        """Add a step to the sequence.

        Args:
            duration: Duration of this step (seconds)
            target_head_pose: Target head pose (None = hold current)
            target_antennas: Target antenna positions (None = hold current)
            target_body_yaw: Target body yaw radians (None = hold current)
        """
        # Use current values if targets not specified
        if target_head_pose is None:
            target_head_pose = self.current_head_pose.copy()
        if target_antennas is None:
            target_antennas_arr = self.current_antennas.copy()
        else:
            target_antennas_arr = np.array(target_antennas)
        if target_body_yaw is None:
            target_body_yaw = self.current_body_yaw

        self.steps.append((duration, target_head_pose, target_antennas_arr, target_body_yaw))
        self._step_start_times.append(self._total_duration)
        self._total_duration += duration

        # Update current values for next step
        self.current_head_pose = target_head_pose
        self.current_antennas = target_antennas_arr
        self.current_body_yaw = target_body_yaw

    @property
    def duration(self) -> float:
        """Total duration of the sequence."""
        return self._total_duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate sequence at time t."""
        if t >= self._total_duration:
            # Sequence complete - hold final pose
            if not self._sequence_complete:
                self._sequence_complete = True
                logger.info(f"Sequence {self.__class__.__name__} complete")

            final_step = self.steps[-1] if self.steps else None
            if final_step:
                _, head, antennas, body_yaw = final_step
                return (head.copy(), antennas.copy(), body_yaw)
            return (None, None, None)

        # Find current step
        step_index = 0
        for i, start_time in enumerate(self._step_start_times):
            if t >= start_time:
                step_index = i

        if step_index != self._current_step_index:
            self._current_step_index = step_index
            logger.debug(f"Sequence step {step_index + 1}/{len(self.steps)}")

        # Get step parameters
        step_duration, target_head, target_antennas, target_body_yaw = self.steps[step_index]
        step_start_time = self._step_start_times[step_index]
        step_local_time = t - step_start_time

        # Get previous step (or initial values)
        if step_index == 0:
            # First step - interpolate from initial pose
            prev_head = self.initial_head_pose
            prev_antennas = self.initial_antennas
            prev_body_yaw = self.initial_body_yaw
        else:
            _, prev_head, prev_antennas, prev_body_yaw = self.steps[step_index - 1]

        # Interpolate within current step with ease-in-out
        interp_t_linear = min(1.0, step_local_time / step_duration) if step_duration > 0 else 1.0
        interp_t = ease_in_out(interp_t_linear)

        # Interpolate head pose
        head_pose = linear_pose_interpolation(prev_head, target_head, interp_t)

        # Interpolate antennas
        antennas = ((1 - interp_t) * prev_antennas + interp_t * target_antennas).astype(np.float64)

        # Interpolate body yaw
        body_yaw = (1 - interp_t) * prev_body_yaw + interp_t * target_body_yaw

        return (head_pose, antennas, body_yaw)


# ============================================================================
# Pout Exit Sequences
# ============================================================================

def create_pout_exit_lunge_sequence(
    current_head_pose: NDArray[np.float32],
    current_antennas: Tuple[float, float],
    current_body_yaw: float,
    user_face_yaw: float | None = None,
) -> SequenceMove:
    """Create 'quit being a baby' lunge sequence.

    Sequence steps:
    1. Exit pout pose → neutral (2s)
    2. Tilt head down, point antennas at user (1s)
    3. Lunge forward aggressively (0.5s)
    4. Hold aggressive posture (1s)
    5. Return to neutral (1s)

    Args:
        current_head_pose: Current head pose
        current_antennas: Current antenna positions
        current_body_yaw: Current body yaw (radians)
        user_face_yaw: User's position in radians (from face detection, optional)

    Returns:
        SequenceMove configured for lunge sequence
    """
    seq = SequenceMove(current_head_pose, current_antennas, current_body_yaw)

    # Neutral position for reference
    neutral_pose = create_head_pose(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, degrees=True, mm=False)

    # Step 1: Exit pout → neutral (2s)
    seq.add_step(
        duration=2.0,
        target_head_pose=neutral_pose,
        target_antennas=(0.0, 0.0),
        target_body_yaw=0.0,
    )

    # Step 2: Tilt head down, antennas point at user (1s)
    # If user position known, point at them; otherwise point straight ahead
    target_yaw = user_face_yaw if user_face_yaw is not None else 0.0
    down_tilt_pose = create_head_pose(
        x=0.0,
        y=0.0,
        z=0.01,
        roll=0.0,
        pitch=-15.0,  # Tilt down 15° (menacing look)
        yaw=np.rad2deg(target_yaw),
        degrees=True,
        mm=False
    )
    seq.add_step(
        duration=1.0,
        target_head_pose=down_tilt_pose,
        target_antennas=(0.0, 0.0),  # Straight up (alert/aggressive)
        target_body_yaw=target_yaw,
    )

    # Step 3: Lunge forward (0.5s)
    lunge_pose = create_head_pose(
        x=0.03,  # 3cm forward lunge
        y=0.0,
        z=0.01,
        roll=0.0,
        pitch=-15.0,  # Maintain downward tilt
        yaw=np.rad2deg(target_yaw),
        degrees=True,
        mm=False
    )
    seq.add_step(
        duration=0.5,
        target_head_pose=lunge_pose,
        target_antennas=(0.0, 0.0),
        target_body_yaw=target_yaw,
    )

    # Step 4: Hold aggressive posture (1s)
    seq.add_step(
        duration=1.0,
        target_head_pose=lunge_pose,  # Hold lunge
        target_antennas=(0.0, 0.0),
        target_body_yaw=target_yaw,
    )

    # Step 5: Return to neutral (1s)
    seq.add_step(
        duration=1.0,
        target_head_pose=neutral_pose,
        target_antennas=(0.0, 0.0),
        target_body_yaw=0.0,
    )

    logger.info("Created pout exit lunge sequence (5 steps, 5.5s total)")
    return seq


def create_pout_exit_gentle_sequence(
    current_head_pose: NDArray[np.float32],
    current_antennas: Tuple[float, float],
    current_body_yaw: float,
    user_face_yaw: float | None = None,
    exit_duration: float = 3.0,
) -> SequenceMove:
    """Create 'i'm sorry laura' gentle forgiveness sequence.

    Sequence steps:
    1. Exit from pout → neutral (configurable duration)
    2. Gentle head tilt toward user (1s)
    3. Return to neutral (1s)

    Args:
        current_head_pose: Current head pose
        current_antennas: Current antenna positions
        current_body_yaw: Current body yaw (radians)
        user_face_yaw: User's position in radians (from face detection, optional)
        exit_duration: Duration for initial exit from pout (0.5=instant, 2.0=normal, 3.0=slow)

    Returns:
        SequenceMove configured for gentle exit sequence
    """
    seq = SequenceMove(current_head_pose, current_antennas, current_body_yaw)

    neutral_pose = create_head_pose(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, degrees=True, mm=False)

    # Step 1: Exit from pout → neutral (configurable speed)
    seq.add_step(
        duration=exit_duration,
        target_head_pose=neutral_pose,
        target_antennas=(0.0, 0.0),
        target_body_yaw=0.0,
    )

    # Step 2: Gentle tilt toward user (1s, acknowledging apology)
    target_yaw = user_face_yaw if user_face_yaw is not None else 0.0
    gentle_look = create_head_pose(
        x=0.0,
        y=0.0,
        z=0.01,
        roll=0.0,
        pitch=5.0,  # Slight upward tilt (less aggressive)
        yaw=np.rad2deg(target_yaw),
        degrees=True,
        mm=False
    )
    seq.add_step(
        duration=1.0,
        target_head_pose=gentle_look,
        target_antennas=(-np.deg2rad(5), np.deg2rad(5)),  # Slight sway (curious)
        target_body_yaw=target_yaw,
    )

    # Step 3: Return to neutral (1s)
    seq.add_step(
        duration=1.0,
        target_head_pose=neutral_pose,
        target_antennas=(0.0, 0.0),
        target_body_yaw=0.0,
    )

    total_duration = exit_duration + 2.0  # exit_duration + 1s + 1s
    logger.info(f"Created pout exit gentle sequence (3 steps, {total_duration}s total, first step: {exit_duration}s)")
    print(f"[POUT EXIT] Sequence created: {len(seq.steps)} steps, total duration: {seq.duration}s")
    return seq
