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
        # Extract current XYZ from start pose - only interpolate rotation to pitch=-10°
        # This prevents jerky startup from XYZ position changes
        current_xyz = interpolation_start_pose[:3, 3]
        self.neutral_head_pose = create_head_pose(
            x=current_xyz[0],
            y=current_xyz[1],
            z=current_xyz[2],
            roll=0.0,
            pitch=-10.0,  # Look up slightly (matches camera neutral)
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

            # Roll tilt - suppressed when strain exceeds threshold
            if self.suppress_due_to_strain:
                roll_tilt = 0.0
            else:
                roll_tilt = -self.roll_amplitude * scale * recovery_scale * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)

            # Y sway - only when anchored to prevent camera motion feedback
            if self.is_anchored:
                # Y sway (side-to-side) - body-relative when anchored
                # Rotate by body anchor yaw so sway stays perpendicular to body orientation
                world_sway = self.sway_amplitude * scale * recovery_scale * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)

                # Rotate sway by body anchor yaw to make it body-relative
                # body_anchor_yaw is already in radians (set by MovementManager)
                y_sway = world_sway * np.cos(self.body_anchor_yaw)  # Y component after rotation
            else:
                y_sway = 0.0

            # Debug actual values once (commented out for cleaner logs)
            # if not hasattr(self, '_sway_debug_printed'):
            #     print(f"BREATHING: roll_tilt={np.rad2deg(roll_tilt):.3f}°, y_sway={y_sway*1000:.3f}mm, scale={scale:.2f}, roll_amp={np.rad2deg(self.roll_amplitude):.2f}°, sway_amp={self.sway_amplitude*1000:.2f}mm", flush=True)
            #     self._sway_debug_printed = True

            # Create breathing pose with body-relative sway (only when anchored)
            neutral_xyz = self.neutral_head_pose[:3, 3]
            head_pose = create_head_pose(
                x=neutral_xyz[0],      # No X sway (could add if needed)
                y=neutral_xyz[1] + y_sway,  # Y sway, body-relative, only when anchored
                z=neutral_xyz[2],      # Keep neutral Z (0.015m)
                roll=np.rad2deg(roll_tilt),  # Convert roll from radians to degrees
                pitch=-10.0,           # Maintain neutral pitch (face tracking adds to this)
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
    """Combine primary and secondary full body poses.

    Args:
        primary_pose: (head_pose, antennas, body_yaw) - primary move
        secondary_pose: (head_pose, antennas, body_yaw) - secondary offsets

    Returns:
        Combined full body pose (head_pose, antennas, body_yaw)

    """
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose

    # Combine head poses using compose_world_offset; the secondary pose must be an
    # offset expressed in the world frame (T_off_world) applied to the absolute
    # primary transform (T_abs).
    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)

    # Sum antennas and body_yaw
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

    # Primary move state
    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0

    # Secondary move state (offsets)
    speech_offsets: Tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    face_tracking_offsets: Tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    # Status flags
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        """Update the last activity time."""
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
        """Reset accumulators while keeping the last potential frequency."""
        self.mean = 0.0
        self.m2 = 0.0
        self.min_freq = float("inf")
        self.count = 0


class MovementManager:
    """Coordinate sequential moves, additive offsets, and robot output at 100 Hz.

    Responsibilities:
    - Own a real-time loop that samples the current primary move (if any), fuses
      secondary offsets, and calls `set_target` exactly once per tick.
    - Start an idle `BreathingMove` after `idle_inactivity_delay` when not
      listening and no moves are queued.
    - Expose thread-safe APIs so other threads can enqueue moves, mark activity,
      or feed secondary offsets without touching internal state.

    Timing:
    - All elapsed-time calculations rely on `time.monotonic()` through `self._now`
      to avoid wall-clock jumps.
    - The loop attempts 100 Hz

    Concurrency:
    - External threads communicate via `_command_queue` messages.
    - Secondary offsets are staged via dirty flags guarded by locks and consumed
      atomically inside the worker loop.
    """

    def __init__(
        self,
        current_robot: ReachyMini,
        camera_worker: "Any" = None,
    ):
        """Initialize movement manager."""
        self.current_robot = current_robot
        self.camera_worker = camera_worker

        # Single timing source for durations
        self._now = time.monotonic

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        # Initialize to desired neutral position so robot doesn't jerk at startup
        neutral_pose = create_head_pose(0.0, 0.0, 0.015, 0.0, -10.0, 0.0, degrees=True, mm=False)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)

        # Move queue (primary moves)
        self.move_queue: deque[Move] = deque()

        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency

        # Debug logging rate limiting
        self._last_breathing_debug = 0.0
        self._breathing_debug_interval = 2.0  # seconds between debug messages

        # Pitch tracking for startup diagnostics (10Hz for first 10s)
        self._pitch_tracking_enabled = True
        self._pitch_tracking_start_time: float | None = None
        self._pitch_tracking_duration = 10.0  # Track for 10 seconds
        self._pitch_tracking_interval = 0.1  # 10Hz = every 0.1s
        self._last_pitch_tracking_time = 0.0
        self._pitch_tracking_count = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_listening = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(self.state.last_primary_pose)
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = self._now()
        self._breathing_active = False  # true when breathing move is running or queued
        self._breathing_paused_external = False  # true when external coordinator pauses breathing
        self._breathing_pause_lock = threading.Lock()

        # Current breathing roll (for camera synchronization)
        self._current_breathing_roll = 0.0  # Current roll angle in radians
        self._breathing_roll_lock = threading.Lock()

        # Breathing amplitude scaling for continuous breathing during external moves
        self._breathing_amplitude_scale = 1.0  # Current amplitude scale (0.0-1.0)
        self._breathing_amplitude_target = 1.0  # Target amplitude for smooth transitions
        self._amplitude_transition_start = 0.0  # Time when transition started
        self._amplitude_transition_duration = 0.0  # Duration of current transition

        # Face tracking ramp-up to prevent jerky activation
        self._face_tracking_scale = 0.0  # Current face tracking scale (0.0-1.0)
        self._face_tracking_ramp_start = 0.0  # Time when ramp started
        self._face_tracking_ramp_duration = 2.0  # Ramp up over 2 seconds
        self._face_tracking_was_suppressed = True  # Track previous suppression state
        self._listening_debounce_s = 0.15
        self._last_listening_toggle_time = self._now()
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 1.0  # seconds between error logs
        self._set_target_err_suppressed = 0

        # Face tracking suppression for primary moves
        self._suppress_face_tracking = False

        # External control flag (for Claude Code mood plugin coordination)
        self._external_control_active = False
        self._external_control_lock = threading.Lock()

        # Body follow state tracking (legacy - now using anchor system)
        self._last_head_yaw_deg = 0.0
        self._head_stable_since = None  # Time when head became stable
        self._body_follow_threshold_deg = 20.0
        self._head_stability_threshold_deg = 5.0  # Max change to be considered stable
        self._body_follow_duration = 2.0  # Duration for smooth body follow interpolation (2 seconds)
        self._body_follow_deadband_deg = 2.0  # Ignore adjustments smaller than this
        self._body_follow_start_yaw = 0.0  # Starting body yaw for interpolation
        self._body_follow_target_yaw = 0.0  # Target body yaw for interpolation
        self._body_follow_start_time = None  # When current follow motion started

        # Anchor-based body yaw control
        self._anchor_state = AnchorState.ANCHORED
        self._body_anchor_yaw = 0.0              # Current anchor point (temp_zero)
        self._current_body_yaw_deg = 0.0         # Tracked body position during sync
        self._strain_threshold_deg = 13.0        # Strain threshold before body follows head
        self._breathing_suppress_threshold_deg = 10.0  # Suppress breathing earlier to prevent collision
        self._current_strain_deg = 0.0           # Current strain for breathing suppression check
        self._stability_duration_s = 1.5         # 1.5 seconds for head stabilization before anchor lock
        self._stability_threshold_deg = 2.0      # 2 degrees max movement to be considered stable

        # Pitch rate limiting (applied in moves.py after clamp, not in camera_worker)
        self._last_commanded_pitch = 0.0         # Previous commanded pitch for rate limiting
        self._max_pitch_change_per_frame = np.deg2rad(5.0)  # Max 5°/frame (~150°/sec at 30Hz)

        # Oscillation detection and recovery
        self._pitch_direction_changes = 0       # Count of direction reversals
        self._last_pitch_change_sign = 0        # Sign of last change (+1, -1, or 0)
        self._oscillation_recovery_mode = False # In recovery mode flag
        self._oscillation_recovery_start = None # When recovery started
        self._oscillation_recovery_duration = 5.0  # Hold at 0° for 5 seconds
        self._oscillation_threshold = 2         # Max direction changes before recovery

        # Cross-thread signalling
        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()
        self._speech_offsets_lock = threading.Lock()
        self._pending_speech_offsets: Tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._speech_offsets_dirty = False

        self._face_offsets_lock = threading.Lock()
        self._pending_face_offsets: Tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._face_offsets_dirty = False

        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = self._is_listening
        self._status_lock = threading.Lock()
        self._freq_stats = LoopFrequencyStats()
        self._freq_snapshot = LoopFrequencyStats()

    def queue_move(self, move: Move) -> None:
        """Queue a primary move to run after the currently executing one.

        Thread-safe: the move is enqueued via the worker command queue so the
        control loop remains the sole mutator of movement state.
        """
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        """Stop the active move and discard any queued primary moves.

        Thread-safe: executed by the worker thread via the command queue.
        """
        self._command_queue.put(("clear_queue", None))

    def goto_sleep_position(self, body_yaw_deg: float = 0.0, duration: float = 2.0) -> None:
        """Move to sleeping position with optional body rotation.

        The sleeping position is a fixed head pose (looking down, resting) that can be
        rotated at the base by specifying body_yaw_deg.

        Args:
            body_yaw_deg: Body yaw rotation in degrees (default 0.0 = forward)
                         Positive = rotate left, Negative = rotate right
            duration: Duration to interpolate to sleep position (default 2.0s)

        Thread-safe: queues move via command queue.
        """
        # Define sleeping head pose (looking down, slightly forward)
        sleep_head_pose = create_head_pose(
            x=0.0,      # Neutral X (centered)
            y=0.0,      # Neutral Y (no side offset)
            z=-0.02,    # Slightly lower than neutral (20mm down)
            yaw=0.0,    # Facing forward relative to body
            pitch=-25.0,  # Looking down 25 degrees
            roll=0.0,   # No roll tilt
            degrees=True
        )

        # Sleeping antenna positions (drooped/relaxed)
        sleep_antennas = (
            np.deg2rad(-10.0),  # Left antenna slightly down
            np.deg2rad(-10.0)   # Right antenna slightly down
        )

        # Convert body yaw to radians
        sleep_body_yaw = np.deg2rad(body_yaw_deg)

        # Create a simple Move class for goto positions
        from reachy_mini.motion.move import Move

        class GotoSleepMove(Move):
            """Simple move to interpolate to sleeping position."""

            def __init__(self, target_pose, target_antennas, target_body_yaw, duration):
                self._target_pose = target_pose
                self._target_antennas = target_antennas
                self._target_body_yaw = target_body_yaw
                self._duration = duration
                self._start_pose = None
                self._start_antennas = None
                self._start_body_yaw = None

            @property
            def duration(self) -> float:
                return self._duration

            def evaluate(self, t: float):
                # On first call, capture starting pose
                if self._start_pose is None:
                    # We don't have access to current pose here, so this will be set
                    # by the movement manager when the move starts
                    # For now, return target at end of duration
                    pass

                # Linear interpolation
                alpha = min(t / self._duration, 1.0)

                # If we had start pose, interpolate. Otherwise just return target.
                # The movement manager will handle this properly.
                if t >= self._duration:
                    return self._target_pose, self._target_antennas, self._target_body_yaw
                else:
                    # Return None to let movement manager handle interpolation
                    return None, None, None

        # Create and queue the move
        sleep_move = GotoSleepMove(
            target_pose=sleep_head_pose,
            target_antennas=sleep_antennas,
            target_body_yaw=sleep_body_yaw,
            duration=duration
        )

        self.queue_move(sleep_move)
        logger.info(f"Queued sleep position: body_yaw={body_yaw_deg:.1f}°, duration={duration}s")

    def set_speech_offsets(self, offsets: Tuple[float, float, float, float, float, float]) -> None:
        """Update speech-induced secondary offsets (x, y, z, roll, pitch, yaw).

        Offsets are interpreted as metres for translation and radians for
        rotation in the world frame. Thread-safe via a pending snapshot.
        """
        with self._speech_offsets_lock:
            self._pending_speech_offsets = offsets
            self._speech_offsets_dirty = True

    def set_moving_state(self, duration: float) -> None:
        """Mark the robot as actively moving for the provided duration.

        Legacy hook used by goto helpers to keep inactivity and breathing logic
        aware of manual motions. Thread-safe via the command queue.
        """
        self._command_queue.put(("set_moving_state", duration))

    def is_idle(self) -> bool:
        """Return True when the robot has been inactive longer than the idle delay."""
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time
            listening = self._shared_is_listening

        if listening:
            return False

        return self._now() - last_activity >= self.idle_inactivity_delay

    def set_external_control(self, active: bool) -> None:
        """Set external control flag for coordination with Claude Code mood plugin.

        When active, face tracking is suppressed to allow external control.
        Thread-safe via lock.
        """
        with self._external_control_lock:
            self._external_control_active = active
            logger.info(f"External control {'enabled' if active else 'disabled'} - face tracking {'suppressed' if active else 'resumed'}")

        # If starting external control, stop any active breathing move
        if active and isinstance(self.state.current_move, BreathingMove):
            self._command_queue.put(("clear_queue", None))
            logger.info("Stopped breathing move for external control")

    def get_current_breathing_roll(self) -> float:
        """Get current breathing roll angle for camera synchronization."""
        with self._breathing_roll_lock:
            return self._current_breathing_roll

    def set_listening(self, listening: bool) -> None:
        """Enable or disable listening mode without touching shared state directly.

        While listening:
        - Antenna positions are frozen at the last commanded values.
        - Blending is reset so that upon unfreezing the antennas return smoothly.
        - Idle breathing is suppressed.

        Thread-safe: the change is posted to the worker command queue.
        """
        with self._shared_state_lock:
            if self._shared_is_listening == listening:
                return
        self._command_queue.put(("set_listening", listening))

    def scale_breathing_down(self, target_scale: float = 0.0, duration: float = 0.5) -> None:
        """Scale breathing amplitude down for external move coordination.

        Keeps breathing move active to prevent auto-restart but reduces amplitude to zero
        so it doesn't conflict with external moves.

        Args:
            target_scale: Target amplitude scale (default 0.0 = no motion)
            duration: Transition duration in seconds (default 0.5s)
        """
        with self._breathing_pause_lock:
            self._breathing_paused_external = True

        # Start amplitude scale transition
        self._breathing_amplitude_target = target_scale
        self._amplitude_transition_start = self._now()
        self._amplitude_transition_duration = duration

        # Sync body to anchor BEFORE relinquishing control (via command queue to avoid race)
        self._command_queue.put(("sync_body_to_anchor", None))

        # Give the worker thread time to process the sync command
        time.sleep(0.1)

        # Relinquish control - stop control loop from calling set_target()
        with self._external_control_lock:
            self._external_control_active = True

        # Clear oscillation recovery mode to prevent interference with external moves
        self._oscillation_recovery_mode = False
        self._oscillation_recovery_start = None
        self._pitch_direction_changes = 0

        logger.info(f"Breathing scaled down to {target_scale*100:.0f}% over {duration}s (control relinquished, body synced to anchor: {self._body_anchor_yaw:.1f}°)")

    def scale_breathing_up(self, target_scale: float = 1.0, duration: float = 1.0) -> None:
        """Scale breathing amplitude back up after external move completes.

        Smoothly transitions breathing back to full amplitude.

        Args:
            target_scale: Target amplitude scale (default 1.0 = 100% full breathing)
            duration: Transition duration in seconds (default 1.0s)
        """
        with self._breathing_pause_lock:
            self._breathing_paused_external = False

        # Start amplitude scale transition
        self._breathing_amplitude_target = target_scale
        self._amplitude_transition_start = self._now()
        self._amplitude_transition_duration = duration

        # Reclaim control - resume control loop set_target() calls
        with self._external_control_lock:
            self._external_control_active = False

        logger.info(f"Breathing scaled up to {target_scale*100:.0f}% over {duration}s (control reclaimed)")

    # Legacy method names for backward compatibility
    def pause_breathing(self) -> None:
        """Legacy method - calls scale_breathing_down()."""
        self.scale_breathing_down()

    def resume_breathing(self) -> None:
        """Legacy method - calls scale_breathing_up()."""
        self.scale_breathing_up()

    def _calculate_current_strain(self) -> None:
        """Calculate current head-body strain for breathing suppression.

        Must be called before _update_breathing_amplitude to ensure strain is current.
        """
        # Get last commanded head pose to calculate strain
        if hasattr(self, '_last_commanded_pose'):
            head_pose = self._last_commanded_pose[0]
            head_yaw_rad = self._extract_yaw_from_pose(head_pose)
            head_yaw_deg = np.rad2deg(head_yaw_rad)

            strain = head_yaw_deg - self._body_anchor_yaw
            # Normalize strain to [-180, 180]
            while strain > 180:
                strain -= 360
            while strain < -180:
                strain += 360

            self._current_strain_deg = strain
        else:
            self._current_strain_deg = 0.0

    def _update_breathing_amplitude(self, current_time: float) -> None:
        """Update breathing amplitude scale with smooth transitions.

        Called every control loop tick to smoothly interpolate amplitude between states.
        Updates the breathing move's amplitude_scale property if breathing is active.
        """
        # Check if we're in a transition
        if self._amplitude_transition_duration > 0:
            elapsed = current_time - self._amplitude_transition_start
            if elapsed < self._amplitude_transition_duration:
                # Interpolate from current to target
                progress = elapsed / self._amplitude_transition_duration
                start_scale = self._breathing_amplitude_scale
                self._breathing_amplitude_scale = start_scale + (self._breathing_amplitude_target - start_scale) * progress
            else:
                # Transition complete
                self._breathing_amplitude_scale = self._breathing_amplitude_target
                self._amplitude_transition_duration = 0.0  # Clear transition

        # Update the breathing move's amplitude and body anchor if it exists
        if isinstance(self.state.current_move, BreathingMove):
            self.state.current_move.amplitude_scale = self._breathing_amplitude_scale
            # Pass current body anchor yaw for coordinate transformation
            self.state.current_move.body_anchor_yaw = np.deg2rad(self._body_anchor_yaw)

            # Suppress roll tilt only when strain exceeds threshold (prevents collision)
            # Sway suppression is handled separately via is_anchored flag
            suppress_roll = abs(self._current_strain_deg) > self._breathing_suppress_threshold_deg
            self.state.current_move.suppress_due_to_strain = suppress_roll

            # Pass anchored state for sway control
            # Only apply sway when anchored to prevent camera motion artifacts
            is_anchored = (self._anchor_state == AnchorState.ANCHORED)
            self.state.current_move.is_anchored = is_anchored

            # Debug once (commented out for cleaner logs)
            # if not hasattr(self, '_anchor_debug_printed'):
            #     print(f"MANAGER_DEBUG: anchor_state={self._anchor_state}, is_anchored={is_anchored}, strain={self._current_strain_deg:.1f}°", flush=True)
            #     self._anchor_debug_printed = True

    def _apply_pitch_rate_limiting(self, pose: FullBodyPose, current_time: float) -> FullBodyPose:
        """Apply pitch rate limiting and oscillation detection to composed pose.

        Args:
            pose: Composed full body pose (head, antennas, body_yaw)
            current_time: Current time for oscillation detection

        Returns:
            Modified pose with rate-limited pitch
        """
        from scipy.spatial.transform import Rotation as R_scipy

        head, antennas, body_yaw = pose

        # Extract current pitch from composed head pose
        R_head = R_scipy.from_matrix(head[:3, :3])
        roll, pitch, yaw = R_head.as_euler("xyz", degrees=False)

        # Clamp pitch to mechanical limits
        max_pitch_up = np.deg2rad(15.0)
        max_pitch_down = np.deg2rad(-15.0)
        clamped_pitch = np.clip(pitch, max_pitch_down, max_pitch_up)

        # Check if in oscillation recovery mode (only active when breathing amplitude > 0.8)
        if self._oscillation_recovery_mode:
            elapsed = current_time - self._oscillation_recovery_start

            if elapsed >= self._oscillation_recovery_duration:
                # Recovery complete - resume normal operation
                self._oscillation_recovery_mode = False
                self._pitch_direction_changes = 0
                self._last_pitch_change_sign = 0
                logger.info("Oscillation recovery complete")
                # Use clamped pitch to resume smoothly
                rate_limited_pitch = clamped_pitch
                self._last_commanded_pitch = rate_limited_pitch
            else:
                # Hold at 0° during recovery
                rate_limited_pitch = 0.0
                self._last_commanded_pitch = 0.0
        else:
            # No rate limiting - CameraWorker handles smooth interpolation
            # Just use clamped pitch directly
            rate_limited_pitch = clamped_pitch
            pitch_change = clamped_pitch - self._last_commanded_pitch

            # Oscillation detection - only when breathing amplitude > 0.8 (near full breathing)
            # This prevents false positives during external moves when amplitude is scaled down
            if not self._external_control_active and self._breathing_amplitude_scale > 0.8:
                if abs(pitch_change) > np.deg2rad(1.0):
                    current_sign = 1 if pitch_change > 0 else -1

                    if self._last_pitch_change_sign != 0 and current_sign != self._last_pitch_change_sign:
                        self._pitch_direction_changes += 1

                        if self._pitch_direction_changes > self._oscillation_threshold:
                            self._oscillation_recovery_mode = True
                            self._oscillation_recovery_start = self._now()
                            self._pitch_direction_changes = 0
                            logger.warning(f"Pitch oscillation detected! Entering recovery mode (0° for {self._oscillation_recovery_duration}s)")
                            rate_limited_pitch = 0.0

                    self._last_pitch_change_sign = current_sign

            self._last_commanded_pitch = rate_limited_pitch

        # Reconstruct rotation with rate-limited pitch
        R_limited = R_scipy.from_euler("xyz", [roll, rate_limited_pitch, yaw], degrees=False)

        # Create modified head pose
        limited_head = np.eye(4, dtype=np.float32)
        limited_head[:3, :3] = R_limited.as_matrix().astype(np.float32)
        limited_head[:3, 3] = head[:3, 3]  # Keep translation

        return (limited_head, antennas, body_yaw)

    def _poll_signals(self, current_time: float) -> None:
        """Apply queued commands and pending offset updates."""
        self._apply_pending_offsets()

        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)

    def _apply_pending_offsets(self) -> None:
        """Apply the most recent speech/face offset updates."""
        speech_offsets: Tuple[float, float, float, float, float, float] | None = None
        with self._speech_offsets_lock:
            if self._speech_offsets_dirty:
                speech_offsets = self._pending_speech_offsets
                self._speech_offsets_dirty = False

        if speech_offsets is not None:
            self.state.speech_offsets = speech_offsets
            self.state.update_activity()
            logger.info(f"[DEBUG] Speech offsets updated, activity reset")

        face_offsets: Tuple[float, float, float, float, float, float] | None = None
        with self._face_offsets_lock:
            if self._face_offsets_dirty:
                face_offsets = self._pending_face_offsets
                self._face_offsets_dirty = False

        if face_offsets is not None:
            self.state.face_tracking_offsets = face_offsets
            # Face tracking is secondary motion - don't reset activity timer
            logger.info(f"[DEBUG] Face offsets updated (no activity reset)")

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        """Handle a single cross-thread command."""
        if command == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                duration = getattr(payload, "duration", None)
                if duration is not None:
                    try:
                        duration_str = f"{float(duration):.2f}"
                    except (TypeError, ValueError):
                        duration_str = str(duration)
                else:
                    duration_str = "?"
                logger.debug(
                    "Queued move with duration %ss, queue size: %s",
                    duration_str,
                    len(self.move_queue),
                )
            else:
                logger.warning("Ignored queue_move command with invalid payload: %s", payload)
        elif command == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            self._suppress_face_tracking = False  # Resume face tracking when queue cleared
            logger.info("Cleared move queue and stopped current move")
        elif command == "set_moving_state":
            try:
                duration = float(payload)
            except (TypeError, ValueError):
                logger.warning("Invalid moving state duration: %s", payload)
                return
            self.state.update_activity()
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired_state = bool(payload)
            now = self._now()
            if now - self._last_listening_toggle_time < self._listening_debounce_s:
                return
            self._last_listening_toggle_time = now

            if self._is_listening == desired_state:
                return

            self._is_listening = desired_state
            self._last_listening_blend_time = now
            if desired_state:
                # Freeze: snapshot current commanded antennas and reset blend
                self._listening_antennas = (
                    float(self._last_commanded_pose[1][0]),
                    float(self._last_commanded_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            else:
                # Unfreeze: restart blending from frozen pose
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()
        elif command == "sync_body_to_anchor":
            # Sync body yaw to current anchor orientation for external control
            try:
                anchor_yaw_rad = np.deg2rad(self._body_anchor_yaw)
                # Get current head and antenna positions to avoid moving them
                _, current_joints = self.current_robot.get_current_joint_positions()
                current_head_pose = self.current_robot.get_current_head_pose()

                if len(current_joints) >= 2:
                    if len(current_joints) == 3:
                        current_antennas = (current_joints[1], current_joints[2])
                    else:
                        current_antennas = (current_joints[0], current_joints[1])

                    # Only move body yaw, keep head and antennas at current positions
                    self.current_robot.set_target(
                        head=current_head_pose,
                        antennas=current_antennas,
                        body_yaw=anchor_yaw_rad
                    )
                    logger.debug(f"Synced body yaw to anchor: {self._body_anchor_yaw:.1f}°")
            except Exception as e:
                logger.warning(f"Failed to sync body to anchor: {e}")
        else:
            logger.warning("Unknown command received by MovementManager: %s", command)

    def _publish_shared_state(self) -> None:
        """Expose idle-related state for external threads."""
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    def _manage_move_queue(self, current_time: float) -> None:
        """Manage the primary move queue (sequential execution)."""
        if self.state.current_move is None or (
            self.state.move_start_time is not None
            and current_time - self.state.move_start_time >= self.state.current_move.duration
        ):
            self.state.current_move = None
            self.state.move_start_time = None
            # Clear face tracking suppression when move completes
            self._suppress_face_tracking = False

            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                # Any real move cancels breathing mode flag
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)
                # Suppress face tracking for primary moves (except BreathingMove)
                self._suppress_face_tracking = not isinstance(self.state.current_move, BreathingMove)
                logger.debug(f"Starting new move, duration: {self.state.current_move.duration}s, face tracking suppressed: {self._suppress_face_tracking}")

    def _manage_breathing(self, current_time: float) -> None:
        """Manage automatic breathing when idle."""
        # Check external control flag
        with self._external_control_lock:
            external_control = self._external_control_active

        # Check external breathing pause flag
        with self._breathing_pause_lock:
            breathing_paused = self._breathing_paused_external

        # Calculate idle time for both debugging and logic
        idle_for = current_time - self.state.last_activity_time

        # Periodic debug (every 2 seconds) - show breathing state
        if current_time - self._last_breathing_debug >= self._breathing_debug_interval:
            self._last_breathing_debug = current_time

            # Only log when breathing is NOT active (avoid spam when already breathing)
            if not self._breathing_active:
                conditions = {
                    "no_move": self.state.current_move is None,
                    "queue_empty": not self.move_queue,
                    "not_listening": not self._is_listening,
                    "no_external": not external_control,
                    "not_paused": not breathing_paused,
                }

                blocking = [k for k, v in conditions.items() if not v]
                idle_ready = idle_for >= self.idle_inactivity_delay

                if blocking or not idle_ready:
                    status = "idle_time" if not idle_ready and not blocking else blocking
                    logger.info(f"Breathing check: idle={idle_for:.2f}s/{self.idle_inactivity_delay}s, blocking={status}")

        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._breathing_active
            and not external_control
            and not breathing_paused
        ):
            if idle_for >= self.idle_inactivity_delay:
                # Set breathing active IMMEDIATELY to prevent repeated attempts
                self._breathing_active = True
                try:
                    # Get robot's actual current physical position
                    _, current_joints = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()

                    # Joint array should be [body_yaw, left_antenna, right_antenna] but may only have antennas
                    if len(current_joints) >= 2:
                        # Only start breathing if robot is fully initialized (need at least antennas)
                        if len(current_joints) == 3:
                            current_antennas = (current_joints[1], current_joints[2])
                        else:
                            # Only antenna values returned
                            current_antennas = (current_joints[0], current_joints[1])

                        self.state.update_activity()

                        # Check if current pose is default (0,0,0) - if so, use desired neutral directly
                        from scipy.spatial.transform import Rotation as R_scipy
                        current_xyz = current_head_pose[:3, 3]
                        current_rotation = R_scipy.from_matrix(current_head_pose[:3, :3])
                        current_rpy = current_rotation.as_euler("xyz", degrees=True)

                        is_default = (np.allclose(current_xyz, 0.0, atol=0.001) and
                                     np.allclose(current_rpy, 0.0, atol=1.0))

                        if is_default:
                            logger.info("Starting pose is default (0,0,0) - using neutral (0,0,0.015,-10°)")
                            # Replace default with desired neutral position
                            from reachy_mini.utils import create_head_pose
                            start_pose = create_head_pose(
                                x=0.0, y=0.0, z=0.015,
                                roll=0.0, pitch=-10.0, yaw=0.0,
                                degrees=True, mm=False
                            )
                            interpolation_duration = 0.0  # No interpolation needed
                        else:
                            logger.info(f"Starting from pose: xyz={current_xyz}, rpy={current_rpy}")
                            start_pose = current_head_pose
                            interpolation_duration = 3.0  # Interpolate to neutral

                        breathing_move = BreathingMove(
                            interpolation_start_pose=start_pose,
                            interpolation_start_antennas=current_antennas,
                            interpolation_duration=interpolation_duration,
                        )
                        self.move_queue.append(breathing_move)
                        logger.info("Started breathing after %.1fs of inactivity", idle_for)
                except Exception as e:
                    self._breathing_active = False
                    logger.error("Failed to start breathing: %s", e)

        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            # Note: face tracking suppression will be set by the new move in _manage_move_queue
            logger.debug("Stopping breathing due to new move activity")

        if self.state.current_move is not None and not isinstance(self.state.current_move, BreathingMove):
            self._breathing_active = False

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from current move or neutral."""
        # When a primary move is playing, sample it and cache the resulting pose
        if self.state.current_move is not None and self.state.move_start_time is not None:
            # Skip breathing move evaluation when external control is active
            with self._external_control_lock:
                external_active = self._external_control_active

            # Pause breathing during external control or body syncing
            suppress_breathing = (
                (isinstance(self.state.current_move, BreathingMove) and external_active) or
                (isinstance(self.state.current_move, BreathingMove) and self._anchor_state == AnchorState.SYNCING)
            )

            if suppress_breathing:
                # Return neutral pose with z=0.01 lift to maintain "alive" appearance
                head = create_head_pose(0, 0, 0.01, 0, 0, 0, degrees=False, mm=False)
                antennas = np.array([0.0, 0.0])
                body_yaw = 0.0
            else:
                move_time = current_time - self.state.move_start_time
                head, antennas, body_yaw = self.state.current_move.evaluate(move_time)

            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([0.0, 0.0])
            if body_yaw is None:
                # Preserve last commanded body_yaw (includes body_follow calculation)
                # Breathing doesn't care about body orientation - let body_follow manage it
                body_yaw = self._last_commanded_pose[2]

            antennas_tuple = (float(antennas[0]), float(antennas[1]))
            head_copy = head.copy()

            primary_full_body_pose = (
                head_copy,
                antennas_tuple,
                float(body_yaw),
            )

            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)
        # Otherwise reuse the last primary pose so we avoid jumps between moves
        elif self.state.last_primary_pose is not None:
            primary_full_body_pose = clone_full_body_pose(self.state.last_primary_pose)
        else:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            primary_full_body_pose = (neutral_head_pose, (0.0, 0.0), 0.0)
            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)

        return primary_full_body_pose


    def _extract_yaw_from_pose(self, head_pose: NDArray[np.float32]) -> float:
        """Extract yaw angle in radians from 4x4 transformation matrix."""
        # Extract rotation matrix (top-left 3x3)
        R = head_pose[:3, :3]
        # Calculate yaw from rotation matrix (assuming ZYX euler convention)
        # yaw = atan2(R[1,0], R[0,0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return yaw

    def _apply_body_follow(self, pose: FullBodyPose) -> FullBodyPose:
        """Apply anchor-based body yaw control with strain threshold.

        State machine:
        - ANCHORED: Body locked at anchor point until strain exceeds threshold
        - SYNCING: Body smoothly interpolating to match head position
        - STABILIZING: Waiting for head to stabilize before setting new anchor

        Uses EASE_IN_OUT for normal tracking, CARTOON during external plugin control
        for more expressive movements.

        Args:
            pose: Full body pose (head_pose, antennas, body_yaw)

        Returns:
            Adjusted full body pose with anchor-based body yaw control
        """
        head_pose, antennas, body_yaw = pose

        # Extract current head yaw (absolute)
        head_yaw_rad = self._extract_yaw_from_pose(head_pose)
        head_yaw_deg = np.rad2deg(head_yaw_rad)
        body_yaw_deg = np.rad2deg(body_yaw)

        now = self._now()

        # Calculate strain (difference between head and anchor point)
        strain = head_yaw_deg - self._body_anchor_yaw

        # Normalize strain to [-180, 180]
        while strain > 180:
            strain -= 360
        while strain < -180:
            strain += 360

        # Store strain for breathing suppression check
        self._current_strain_deg = strain

        # Debug logging counter (unused but kept for compatibility)
        if not hasattr(self, '_body_follow_log_counter'):
            self._body_follow_log_counter = 0
        self._body_follow_log_counter += 1

        # STATE MACHINE
        if self._anchor_state == AnchorState.ANCHORED:
            # Check if strain exceeds threshold
            if abs(strain) > self._strain_threshold_deg:
                # Trigger sync - capture current body position
                self._anchor_state = AnchorState.SYNCING
                self._body_follow_start_time = now
                self._current_body_yaw_deg = self._body_anchor_yaw  # Start from current anchor
            else:
                # Stay anchored - lock body at anchor point
                self._current_body_yaw_deg = self._body_anchor_yaw
                body_yaw_deg = self._body_anchor_yaw

        elif self._anchor_state == AnchorState.SYNCING:
            # Body continuously tracks head's CURRENT position (not stale captured target)
            # Apply rate limiting to smooth the movement
            max_change_per_tick = 90.0 / self._body_follow_duration / 100.0  # degrees per tick at 100Hz

            desired_change = head_yaw_deg - self._current_body_yaw_deg
            # Normalize to [-180, 180]
            while desired_change > 180:
                desired_change -= 360
            while desired_change < -180:
                desired_change += 360

            # Clamp change to max rate
            actual_change = np.clip(desired_change, -max_change_per_tick, max_change_per_tick)
            self._current_body_yaw_deg = self._current_body_yaw_deg + actual_change
            body_yaw_deg = self._current_body_yaw_deg

            # Check if body caught up with head (within 3°)
            if abs(head_yaw_deg - self._current_body_yaw_deg) < 3.0:
                logger.debug("Anchor: Body caught up with head, entering stabilization phase")
                self._anchor_state = AnchorState.STABILIZING
                self._head_stable_since = None

        elif self._anchor_state == AnchorState.STABILIZING:
            # Body matches head, wait for stability
            self._current_body_yaw_deg = head_yaw_deg
            body_yaw_deg = head_yaw_deg

            # Track head movement
            head_change = abs(head_yaw_deg - self._last_head_yaw_deg)

            if head_change > self._stability_threshold_deg:
                # Head moved significantly, reset stability timer
                self._head_stable_since = None
            elif self._head_stable_since is None:
                # Head just became stable, start timer
                self._head_stable_since = now
            else:
                # Check if stable long enough
                stable_duration = now - self._head_stable_since
                if stable_duration >= self._stability_duration_s:
                    # Establish new anchor!
                    logger.debug(f"Anchor: Head stable for {stable_duration:.1f}s, setting new anchor at {head_yaw_deg:.1f}°")
                    self._body_anchor_yaw = head_yaw_deg
                    self._anchor_state = AnchorState.ANCHORED

        # Update last head yaw for stability tracking
        self._last_head_yaw_deg = head_yaw_deg

        return (head_pose, antennas, np.deg2rad(body_yaw_deg))

    def _compose_full_body_pose(self, current_time: float) -> FullBodyPose:
        """Compose primary and secondary poses into a single command pose.

        Restored to original elegant design:
        - Primary: Breathing or explicit move (continuous base layer)
        - Secondary: Face tracking + speech sway (additive offsets)
        - Composition: combine_full_body(primary, secondary)
        - Safety layers: pitch rate limiting → body follow

        Face tracking and breathing work TOGETHER via additive composition.
        """
        # Get primary pose (breathing with scaled amplitude, or other move)
        primary = self._get_primary_pose(current_time)

        # Get secondary pose (face tracking + speech sway offsets)
        secondary = self._get_secondary_pose()

        # PROPER DECOMPOSITION to prevent roll from becoming yaw
        # Face tracking controls yaw/pitch, breathing adds roll
        # Must decompose and recompose to keep them separate
        primary_head, primary_antennas, primary_body_yaw = primary
        secondary_head, secondary_antennas, secondary_body_yaw = secondary

        from scipy.spatial.transform import Rotation as R_scipy

        # Extract face tracking yaw/pitch from secondary
        R_face = R_scipy.from_matrix(secondary_head[:3, :3])
        _, face_pitch, face_yaw = R_face.as_euler("xyz", degrees=False)

        # Extract breathing roll from primary
        R_breathing = R_scipy.from_matrix(primary_head[:3, :3])
        breathing_roll, _, _ = R_breathing.as_euler("xyz", degrees=False)

        # Update current breathing roll for camera synchronization
        with self._breathing_roll_lock:
            self._current_breathing_roll = breathing_roll

        # Combine: face tracking yaw/pitch + breathing roll
        # This keeps roll as pure roll regardless of head yaw
        R_combined = R_scipy.from_euler("xyz", [breathing_roll, face_pitch, face_yaw], degrees=False)

        # Use face tracking translation (includes IK z-raise)
        combined_head = np.eye(4, dtype=np.float32)
        combined_head[:3, :3] = R_combined.as_matrix().astype(np.float32)
        combined_head[:3, 3] = secondary_head[:3, 3]  # Face tracking translation

        # Sum antennas and body yaw
        combined_antennas = (
            primary_antennas[0] + secondary_antennas[0],
            primary_antennas[1] + secondary_antennas[1],
        )
        combined_body_yaw = primary_body_yaw + secondary_body_yaw

        composed = (combined_head, combined_antennas, combined_body_yaw)

        # Apply safety layers AFTER composition
        rate_limited = self._apply_pitch_rate_limiting(composed, current_time)
        final = self._apply_body_follow(rate_limited)

        return final

    def _get_secondary_pose(self) -> FullBodyPose:
        """Get the secondary full body pose from speech and face tracking offsets.

        Both face tracking and speech sway are secondary movements that compose
        additively with the primary movement (breathing or explicit move).
        """
        # Combine speech sway offsets + face tracking offsets for secondary pose
        secondary_offsets = [
            self.state.speech_offsets[0] + self.state.face_tracking_offsets[0],
            self.state.speech_offsets[1] + self.state.face_tracking_offsets[1],
            self.state.speech_offsets[2] + self.state.face_tracking_offsets[2],
            self.state.speech_offsets[3] + self.state.face_tracking_offsets[3],
            self.state.speech_offsets[4] + self.state.face_tracking_offsets[4],
            self.state.speech_offsets[5] + self.state.face_tracking_offsets[5],
        ]

        secondary_head_pose = create_head_pose(
            x=secondary_offsets[0],
            y=secondary_offsets[1],
            z=secondary_offsets[2],
            roll=secondary_offsets[3],
            pitch=secondary_offsets[4],
            yaw=secondary_offsets[5],
            degrees=False,
            mm=False,
        )
        return (secondary_head_pose, (0.0, 0.0), 0.0)

    def _update_primary_motion(self, current_time: float) -> None:
        """Advance queue state and idle behaviours for this tick."""
        self._manage_move_queue(current_time)
        self._manage_breathing(current_time)

    def _calculate_blended_antennas(self, target_antennas: Tuple[float, float]) -> Tuple[float, float]:
        """Blend target antennas with listening freeze state and update blending."""
        now = self._now()
        listening = self._is_listening
        listening_antennas = self._listening_antennas
        blend = self._antenna_unfreeze_blend
        blend_duration = self._antenna_blend_duration
        last_update = self._last_listening_blend_time
        self._last_listening_blend_time = now

        if listening:
            antennas_cmd = listening_antennas
            new_blend = 0.0
        else:
            dt = max(0.0, now - last_update)
            if blend_duration <= 0:
                new_blend = 1.0
            else:
                new_blend = min(1.0, blend + dt / blend_duration)
            antennas_cmd = (
                listening_antennas[0] * (1.0 - new_blend) + target_antennas[0] * new_blend,
                listening_antennas[1] * (1.0 - new_blend) + target_antennas[1] * new_blend,
            )

        if listening:
            self._antenna_unfreeze_blend = 0.0
        else:
            self._antenna_unfreeze_blend = new_blend
            if new_blend >= 1.0:
                self._listening_antennas = (
                    float(target_antennas[0]),
                    float(target_antennas[1]),
                )

        return antennas_cmd

    def _issue_control_command(self, head: NDArray[np.float32], antennas: Tuple[float, float], body_yaw: float) -> None:
        """Send the fused pose to the robot with throttled error logging."""
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
        except Exception as e:
            now = self._now()
            if now - self._last_set_target_err >= self._set_target_err_interval:
                msg = f"Failed to set robot target: {e}"
                if self._set_target_err_suppressed:
                    msg += f" (suppressed {self._set_target_err_suppressed} repeats)"
                    self._set_target_err_suppressed = 0
                logger.error(msg)
                self._last_set_target_err = now
            else:
                self._set_target_err_suppressed += 1
        else:
            with self._status_lock:
                self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))

                # Initialize anchor on first successful command
                if self._anchor_state == AnchorState.ANCHORED and self._body_anchor_yaw == 0.0:
                    body_yaw_deg = np.rad2deg(body_yaw)
                    self._body_anchor_yaw = body_yaw_deg
                    logger.debug(f"Anchor: Initialized anchor at {body_yaw_deg:.1f}°")

    def _update_frequency_stats(
        self, loop_start: float, prev_loop_start: float, stats: LoopFrequencyStats,
    ) -> LoopFrequencyStats:
        """Update frequency statistics based on the current loop start time."""
        period = loop_start - prev_loop_start
        if period > 0:
            stats.last_freq = 1.0 / period
            stats.count += 1
            delta = stats.last_freq - stats.mean
            stats.mean += delta / stats.count
            stats.m2 += delta * (stats.last_freq - stats.mean)
            stats.min_freq = min(stats.min_freq, stats.last_freq)
        return stats

    def _schedule_next_tick(self, loop_start: float, stats: LoopFrequencyStats) -> Tuple[float, LoopFrequencyStats]:
        """Compute sleep time to maintain target frequency and update potential freq."""
        computation_time = self._now() - loop_start
        stats.potential_freq = 1.0 / computation_time if computation_time > 0 else float("inf")
        sleep_time = max(0.0, self.target_period - computation_time)
        return sleep_time, stats

    def _record_frequency_snapshot(self, stats: LoopFrequencyStats) -> None:
        """Store a thread-safe snapshot of current frequency statistics."""
        with self._status_lock:
            self._freq_snapshot = LoopFrequencyStats(
                mean=stats.mean,
                m2=stats.m2,
                min_freq=stats.min_freq,
                count=stats.count,
                last_freq=stats.last_freq,
                potential_freq=stats.potential_freq,
            )

    def _maybe_log_frequency(self, loop_count: int, print_interval_loops: int, stats: LoopFrequencyStats) -> None:
        """Emit frequency telemetry when enough loops have elapsed."""
        if loop_count % print_interval_loops != 0 or stats.count == 0:
            return

        variance = stats.m2 / stats.count if stats.count > 0 else 0.0
        lowest = stats.min_freq if stats.min_freq != float("inf") else 0.0
        logger.debug(
            "Loop freq - avg: %.2fHz, variance: %.4f, min: %.2fHz, last: %.2fHz, potential: %.2fHz, target: %.1fHz",
            stats.mean,
            variance,
            lowest,
            stats.last_freq,
            stats.potential_freq,
            self.target_frequency,
        )
        stats.reset()

    def _track_pitch_diagnostics(self, head_pose: NDArray[np.float32], current_time: float) -> None:
        """Track pitch values at 10Hz for first 10 seconds after startup.

        This diagnostic logging helps identify oscillation causes during startup.
        """
        if not self._pitch_tracking_enabled:
            return

        # Pose tracking disabled - was for debugging startup issues

    def _update_face_tracking(self, current_time: float) -> None:
        """Update camera worker with current breathing pose and fetch face tracking data.

        Face tracking offsets are RELATIVE (delta from breathing pose) and compose
        additively with breathing. When suppressed, offsets are set to (0,0,0,0,0,0)
        which means "no offset" in the composition.
        """
        # Check external control flag with thread safety
        with self._external_control_lock:
            external_control = self._external_control_active

        # Face tracking only works with breathing - suppress if no breathing move active
        no_breathing = not isinstance(self.state.current_move, BreathingMove)

        # Check if face tracking is suppressed (primary move OR external control OR no breathing)
        # NO MORE breathing startup suppression - face tracking is immediate
        is_suppressed = self._suppress_face_tracking or external_control or no_breathing

        # Simple face tracking - no ramp-up, no scaling
        # Either fully on or fully off

        if is_suppressed:
            # Use neutral offsets when suppressed (means "no offset" in composition)
            self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        elif self.camera_worker is not None:
            # Get face tracking offsets from camera worker thread - use directly
            offsets = self.camera_worker.get_face_tracking_offsets()
            self.state.face_tracking_offsets = offsets
        else:
            # No camera worker, use neutral offsets
            self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def start(self) -> None:
        """Start the worker thread that drives the 100 Hz control loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Move worker already running; start() ignored")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Move worker started")

    def stop(self) -> None:
        """Request the worker thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.debug("Move worker stopped")

    def get_status(self) -> Dict[str, Any]:
        """Return a lightweight status snapshot for observability."""
        with self._status_lock:
            pose_snapshot = clone_full_body_pose(self._last_commanded_pose)
            freq_snapshot = LoopFrequencyStats(
                mean=self._freq_snapshot.mean,
                m2=self._freq_snapshot.m2,
                min_freq=self._freq_snapshot.min_freq,
                count=self._freq_snapshot.count,
                last_freq=self._freq_snapshot.last_freq,
                potential_freq=self._freq_snapshot.potential_freq,
            )

        head_matrix = pose_snapshot[0].tolist() if pose_snapshot else None
        antennas = pose_snapshot[1] if pose_snapshot else None
        body_yaw = pose_snapshot[2] if pose_snapshot else None

        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
            "last_commanded_pose": {
                "head": head_matrix,
                "antennas": antennas,
                "body_yaw": body_yaw,
            },
            "loop_frequency": {
                "last": freq_snapshot.last_freq,
                "mean": freq_snapshot.mean,
                "min": freq_snapshot.min_freq,
                "potential": freq_snapshot.potential_freq,
                "samples": freq_snapshot.count,
            },
        }

    def working_loop(self) -> None:
        """Control loop main movements - reproduces main_works.py control architecture.

        Single set_target() call with pose fusion.
        """
        logger.info("Starting enhanced movement control loop (100Hz)")

        loop_count = 0
        prev_loop_start = self._now()
        print_interval_loops = max(1, int(self.target_frequency * 2))
        freq_stats = self._freq_stats

        while not self._stop_event.is_set():
            loop_start = self._now()
            loop_count += 1

            if loop_count > 1:
                freq_stats = self._update_frequency_stats(loop_start, prev_loop_start, freq_stats)
            prev_loop_start = loop_start

            # Check external control flag FIRST - skip all processing if external system has control
            with self._external_control_lock:
                external_control = self._external_control_active

            if not external_control:
                # Normal control loop - we have control
                # 1) Poll external commands and apply pending offsets (atomic snapshot)
                self._poll_signals(loop_start)

                # 1.3) Calculate current strain for breathing suppression
                self._calculate_current_strain()

                # 1.5) Update breathing amplitude scaling (smooth transitions)
                self._update_breathing_amplitude(loop_start)

                # 2) Manage the primary move queue (start new move, end finished move, breathing)
                self._update_primary_motion(loop_start)

                # 3) Update vision-based secondary offsets
                self._update_face_tracking(loop_start)

                # 4) Build primary and secondary full-body poses, then fuse them
                head, antennas, body_yaw = self._compose_full_body_pose(loop_start)

                # 5) Apply listening antenna freeze or blend-back
                antennas_cmd = self._calculate_blended_antennas(antennas)

                # 6) Single set_target call
                self._issue_control_command(head, antennas_cmd, body_yaw)

                # 6.5) Pitch tracking diagnostics (10Hz for first 10s)
                self._track_pitch_diagnostics(head, loop_start)
            # else: External system has full control - do nothing, just sleep

            # 7) Adaptive sleep to align to next tick, then publish shared state
            sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
            self._publish_shared_state()
            self._record_frequency_snapshot(freq_stats)

            # 8) Periodic telemetry on loop frequency
            self._maybe_log_frequency(loop_count, print_interval_loops, freq_stats)

            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")
