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

            # DISABLED: Y sway - causing coordinate frame issues
            # Y sway - only when anchored to prevent camera motion feedback
            x_sway = 0.0
            y_sway = 0.0
            # if self.is_anchored:
            #     # Body-relative sway (side-to-side) - rotate to world coordinates
            #     # In body frame: X=0, Y=sway (perpendicular to body orientation)
            #     # Rotate by body anchor yaw to get world X and Y components
            #     body_local_sway = self.sway_amplitude * scale * recovery_scale * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            #
            #     # Transform body-local Y sway into world coordinates
            #     # body_anchor_yaw is already in radians (set by MovementManager)
            #     x_sway = -body_local_sway * np.sin(self.body_anchor_yaw)  # X component in world frame
            #     y_sway = body_local_sway * np.cos(self.body_anchor_yaw)   # Y component in world frame
            # else:
            #     x_sway = 0.0
            #     y_sway = 0.0

            # Create breathing pose with body-relative sway (only when anchored)
            neutral_xyz = self.neutral_head_pose[:3, 3]
            head_pose = create_head_pose(
                x=neutral_xyz[0] + x_sway,  # X component from rotated sway
                y=neutral_xyz[1] + y_sway,  # Y component from rotated sway
                z=neutral_xyz[2],      # Keep neutral Z (0.01m)
                roll=np.rad2deg(roll_tilt),  # Convert roll from radians to degrees
                pitch=0.0,             # Maintain neutral pitch (face tracking adds to this)
                yaw=0.0,               # Neutral yaw (face tracking adds to this)
                degrees=True,   # All angles in degrees
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

    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)

    combined_antennas = (
        primary_antennas[0] + secondary_antennas[0],
        primary_antennas[1] + secondary_antennas[1],
    )
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
        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()
        self._speech_offsets_lock = threading.Lock()
        self._pending_speech_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._speech_offsets_dirty = False
        self._face_offsets_lock = threading.Lock()
        self._pending_face_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._face_offsets_dirty = False
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

    def set_speech_offsets(self, offsets: Tuple[float, float, float, float, float, float]) -> None:
        with self._speech_offsets_lock:
            self._pending_speech_offsets = offsets
            self._speech_offsets_dirty = True

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
                self._speech_offsets_dirty = False
        if self._face_offsets_dirty:
            with self._face_offsets_lock:
                self.state.face_tracking_offsets = self._pending_face_offsets
                self._face_offsets_dirty = False

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        if command == "queue_move":
            self.move_queue.append(payload)
            self.state.update_activity()
        elif command == "clear_queue":
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
        if not self.state.current_move or move_ended:
            self.state.current_move = None
            self._suppress_face_tracking = False
            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)
                self._suppress_face_tracking = not self._breathing_active

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
        head_pose, antennas, body_yaw = pose

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

        if self._anchor_state == AnchorState.ANCHORED:
            if abs(strain) > self._strain_threshold_deg:
                self._anchor_state = AnchorState.SYNCING
                self._current_body_yaw_deg = self._body_anchor_yaw
            body_yaw = np.deg2rad(self._body_anchor_yaw)
        elif self._anchor_state == AnchorState.SYNCING:
            change = (head_yaw_deg - self._current_body_yaw_deg + 180) % 360 - 180
            self._current_body_yaw_deg += np.clip(change, -0.45, 0.45)
            if abs(head_yaw_deg - self._current_body_yaw_deg) < 3.0:
                self._anchor_state = AnchorState.STABILIZING
                self._head_stable_since = None
            body_yaw = np.deg2rad(self._current_body_yaw_deg)
        elif self._anchor_state == AnchorState.STABILIZING:
            self._current_body_yaw_deg = head_yaw_deg
            if abs(head_yaw_deg - self._last_head_yaw_deg) > self._stability_threshold_deg:
                self._head_stable_since = None
            elif self._head_stable_since is None:
                self._head_stable_since = now
            elif now - self._head_stable_since >= self._stability_duration_s:
                self._body_anchor_yaw = head_yaw_deg
                self._anchor_state = AnchorState.ANCHORED
            body_yaw = np.deg2rad(self._current_body_yaw_deg)

        self._last_head_yaw_deg = head_yaw_deg
        return (head_pose, antennas, body_yaw)

    def _compose_full_body_pose(self, current_time: float) -> FullBodyPose:
        primary = self._get_primary_pose(current_time)
        secondary = self._get_secondary_pose()
        composed = combine_full_body(primary, secondary)
        return self._apply_body_follow(composed)

    def _get_secondary_pose(self) -> FullBodyPose:
        offsets = [s + f for s, f in zip(self.state.speech_offsets, self.state.face_tracking_offsets)]
        head_pose = create_head_pose(*offsets, degrees=False, mm=False)
        return (head_pose, (0.0, 0.0), 0.0)

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
            self.state.face_tracking_offsets = (0.0,) * 6
        else:
            self.state.face_tracking_offsets = self.camera_worker.get_face_tracking_offsets()

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
