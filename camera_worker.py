"""Camera worker thread with frame buffering and face tracking.

Ported from main_works.py camera_worker() function to provide:
- 30Hz+ camera polling with thread-safe frame buffering
- Face tracking integration with smooth interpolation
- Latest frame always available for tools
"""

import time
import logging
import threading
from typing import Any, List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini.utils.interpolation import linear_pose_interpolation


logger = logging.getLogger(__name__)


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


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(self, media_manager: Any, head_tracker: Any = None, daemon_client: Any = None, movement_manager: Any = None, debug_window: bool = False) -> None:
        """Initialize.

        Args:
            media_manager: MediaManager instance for camera access
            head_tracker: Optional head tracker for face detection
            daemon_client: DaemonClient instance for IK calculations
            movement_manager: MovementManager instance for breathing synchronization
            debug_window: Show debug visualization window with face detection overlays
        """
        self.media_manager = media_manager
        self.head_tracker = head_tracker
        self.daemon_client = daemon_client
        self.movement_manager = movement_manager
        self.debug_window = debug_window

        # Thread-safe frame storage
        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Face tracking state
        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        # Debug visualization state
        self._last_face_center: Tuple[int, int] | None = None
        self._last_yaw_deg: float = 0.0
        self._last_world_yaw_deg: float = 0.0
        self._last_body_yaw_deg: float = 0.0

        # Face tracking timing variables (same as main_works.py)
        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.face_lost_delay = 10.0  # seconds to wait before starting interpolation (hold position)
        self.interpolation_duration = 2.0  # seconds to interpolate back to neutral

        # Track state changes
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        # Pitch interpolation synchronized with detection frequency (1s)
        self._current_interpolated_pitch = np.deg2rad(0.0)  # Start at neutral (0°)
        self._pitch_interpolation_target: float | None = None  # Target pitch to interpolate toward
        self._pitch_offset = np.deg2rad(5.0)  # Offset to apply to IK pitch (positive = look more down)
        self._pitch_interpolation_start: float | None = None   # Starting pitch of current interpolation
        self._pitch_interpolation_start_time: float | None = None  # When interpolation started
        self._pitch_interpolation_duration = 0.3  # 300ms smooth transition (longer to prevent oscillation)

        # Yaw interpolation synchronized with detection frequency
        self._current_interpolated_yaw = 0.0  # Start at neutral (0°)
        self._yaw_interpolation_target: float | None = None
        self._yaw_interpolation_start: float | None = None
        self._yaw_interpolation_start_time: float | None = None
        self._yaw_interpolation_duration = 0.3  # Match pitch interpolation (300ms)

        # Breathing roll synchronization (sample at zero crossings with hysteresis)
        self._last_breathing_roll = 0.0  # Track previous roll for edge detection
        self._roll_away_from_zero = False  # Start False to require breathing motion before first sample
        self._last_sample_time = time.time()  # Initialize to current time to prevent immediate fallback

        # World-frame target for body-follow (separate from body-relative offsets)
        self._world_frame_target_yaw: float | None = None  # Absolute world yaw to face target
        self._world_frame_target_lock = threading.Lock()

        # Track last pixel position to detect when target moved significantly
        self._last_target_pixel_x: int | None = None
        self._target_movement_threshold_px = 50  # Recalculate world target if moved >50px

        # Track last sampled values for continuous offset updates
        self._last_translation = np.array([0.0, 0.0, 0.0])
        self._last_roll = 0.0
        self._last_yaw = 0.0

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
            frame_rgb: NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[assignment]
            return frame_rgb

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def get_world_frame_target_yaw(self) -> float | None:
        """Get world-frame target yaw for body-follow strain calculation (thread-safe).

        Returns:
            World-frame yaw in radians, or None if no face is being tracked
        """
        with self._world_frame_target_lock:
            return self._world_frame_target_yaw

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking."""
        print(f"[CAMERA WORKER] set_head_tracking_enabled({enabled})")
        self.is_head_tracking_enabled = enabled
        if not enabled:
            # Reset pitch interpolation state when disabling
            self._current_interpolated_pitch = np.deg2rad(0.0)  # Back to neutral
            self._pitch_interpolation_target = None
            self._pitch_interpolation_start = None
            self._pitch_interpolation_start_time = None
            # Reset yaw interpolation state when disabling
            self._current_interpolated_yaw = 0.0  # Back to neutral
            self._yaw_interpolation_target = None
            self._yaw_interpolation_start = None
            self._yaw_interpolation_start_time = None
            # Reset breathing roll synchronization
            self._last_breathing_roll = 0.0
            self._roll_away_from_zero = False
        else:
            # When re-enabling, clear frame buffers and reset state
            print("[CAMERA WORKER] Clearing frame buffers and resetting state")

            # Clear latest frame
            with self.frame_lock:
                self.latest_frame = None

            # Flush media manager frame buffer to get fresh frames (increased from 5 to 30 frames)
            if self.media_manager:
                # Consume and discard many frames to ensure buffer is completely clear
                # At 30 fps, this clears 1 second of buffered frames
                for _ in range(30):
                    try:
                        _ = self.media_manager.get_frame()
                    except:
                        pass

            # Reset hysteresis to require moving away from zero before allowing next sample
            # This prevents immediate sampling when re-enabling
            self._roll_away_from_zero = False  # Start False - must move away before crossing can trigger

            # Set last sample time to current time to prevent fallback timeout from triggering
            self._last_sample_time = time.time()

            # Reset face detection state
            self.last_face_detected_time = None
            self.interpolation_start_time = None

            print(f"[CAMERA WORKER] State reset complete - will wait for breathing to move away from zero before sampling")

        logger.info(f"Head tracking {'enabled' if enabled else 'disabled'}")

    def set_pitch_offset(self, offset_degrees: float) -> None:
        """Set pitch offset in degrees (positive = look more down, negative = look more up)."""
        self._pitch_offset = np.deg2rad(offset_degrees)
        print(f"[PITCH] Offset set to {offset_degrees:.1f}°")


    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        # Close debug window if it was open
        if self.debug_window:
            cv2.destroyAllWindows()

        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Enable the camera worker loop.

        Ported from main_works.py camera_worker() with same logic.
        """
        logger.debug("Starting camera working loop")

        # Initialize head tracker if available
        # Neutral pose: 0° pitch (looking straight) + 1.0cm z-lift
        neutral_pose = np.eye(4, dtype=np.float32)
        neutral_pose[2, 3] = 0.01  # Raise head by 1.0cm to avoid low position problems
        neutral_rotation = R.from_euler("xyz", [0, np.deg2rad(0.0), 0])  # 0° pitch (looking straight)
        neutral_pose[:3, :3] = neutral_rotation.as_matrix()
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        # Move to neutral position on startup
        if self.daemon_client:
            try:
                logger.info("Moving to neutral head position on startup")
                self.daemon_client.set_target(head=neutral_pose)
                time.sleep(0.5)  # Give robot time to reach neutral
            except Exception as e:
                logger.warning(f"Failed to move to neutral position: {e}")

        # Flush stale frames from daemon (clear any frames from previous session)
        logger.debug("Flushing stale frames from camera...")
        for _ in range(90):  # Flush 3 seconds of frames at 30fps to clear daemon buffer
            self.media_manager.get_frame()
        logger.debug("Camera frames flushed, starting fresh")

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # If face tracking disabled, sleep and skip frame capture entirely
                if not self.is_head_tracking_enabled:
                    time.sleep(0.1)  # Sleep 100ms to free CPU during moods
                    continue

                # Get frame from media manager for display
                frame = self.media_manager.get_frame()

                if frame is not None:
                    # Thread-safe frame storage
                    with self.frame_lock:
                        self.latest_frame = frame

                # Check if face tracking was just disabled
                if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                    # Face tracking was just disabled - start interpolation to neutral
                    self.last_face_detected_time = current_time  # Trigger the face-lost logic
                    self.interpolation_start_time = None  # Will be set by the face-lost interpolation
                    self.interpolation_start_pose = None
                    # Reset pitch interpolation to neutral
                    self._current_interpolated_pitch = np.deg2rad(0.0)
                    self._pitch_interpolation_target = None
                    self._pitch_interpolation_start = None
                    self._pitch_interpolation_start_time = None
                    # Reset yaw filter to neutral
                    self._filtered_yaw = 0.0

                # Update tracking state
                self.previous_head_tracking_state = self.is_head_tracking_enabled

                # Handle face tracking if enabled and head tracker available
                if self.head_tracker is not None:
                    # Update pitch interpolation EVERY frame (not just when sampling)
                    if self._pitch_interpolation_target is not None and self._pitch_interpolation_start_time is not None:
                        elapsed = current_time - self._pitch_interpolation_start_time
                        t_linear = min(1.0, elapsed / self._pitch_interpolation_duration)

                        # Linear interpolation (no easing to avoid feedback oscillation)
                        self._current_interpolated_pitch = (
                            self._pitch_interpolation_start * (1.0 - t_linear) +
                            self._pitch_interpolation_target * t_linear
                        )

                        # If interpolation complete, clear state
                        if t_linear >= 1.0:
                            self._pitch_interpolation_target = None
                            self._pitch_interpolation_start = None
                            self._pitch_interpolation_start_time = None

                    # Update yaw interpolation EVERY frame (not just when sampling)
                    if self._yaw_interpolation_target is not None and self._yaw_interpolation_start_time is not None:
                        elapsed = current_time - self._yaw_interpolation_start_time
                        t_linear = min(1.0, elapsed / self._yaw_interpolation_duration)

                        # Linear interpolation (no easing to avoid feedback oscillation)
                        self._current_interpolated_yaw = (
                            self._yaw_interpolation_start * (1.0 - t_linear) +
                            self._yaw_interpolation_target * t_linear
                        )

                        # If interpolation complete, clear state
                        if t_linear >= 1.0:
                            self._yaw_interpolation_target = None
                            self._yaw_interpolation_start = None
                            self._yaw_interpolation_start_time = None

                    # Update face tracking offsets EVERY frame with interpolated pitch AND yaw
                    # Face tracking uses ONLY rotations (roll, pitch, yaw), not translations
                    # The Stewart platform achieves target orientation through rotation alone
                    # ONLY update if NOT in neutral recovery mode (prevents race condition)
                    if self.interpolation_start_time is None:
                        with self.face_tracking_lock:
                            self.face_tracking_offsets = [
                                0.0,  # x translation (not used for face tracking)
                                0.0,  # y translation (not used for face tracking)
                                0.0,  # z translation (not used for face tracking)
                                self._last_roll,
                                self._current_interpolated_pitch,
                                self._current_interpolated_yaw,
                            ]

                    # Time-based sampling: every 0.1 seconds (10 Hz with MediaPipe)
                    should_sample = (current_time - self._last_sample_time) >= 0.1

                    if not should_sample:
                        # Not time to sample yet - skip this frame
                        continue

                    # Update last sample time
                    self._last_sample_time = current_time

                    # Capture a fresh frame for MediaPipe detection
                    detection_frame = self.media_manager.get_frame()
                    if detection_frame is None:
                        continue

                    # Run face detection (fast with MediaPipe - 10 Hz / every 0.1 second)
                    eye_center, _ = self.head_tracker.get_head_position(detection_frame)

                    if eye_center is not None:
                        # Face detected - immediately switch to tracking

                        # If recovering to neutral, capture current position to prevent discontinuity
                        if self.interpolation_start_time is not None:
                            with self.face_tracking_lock:
                                # Extract current pitch/yaw from neutral recovery interpolation
                                self._current_interpolated_pitch = self.face_tracking_offsets[4]
                                self._current_interpolated_yaw = self.face_tracking_offsets[5]

                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None  # Stop any face-lost interpolation

                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = detection_frame.shape
                        eye_center_norm = (eye_center + 1) / 2
                        eye_center_pixels = [
                            eye_center_norm[0] * w,
                            eye_center_norm[1] * h,
                        ]

                        # Get the head pose needed to look at the target via daemon IK
                        if self.daemon_client is None:
                            continue

                        try:
                            u_pixel = int(eye_center_pixels[0])
                            v_pixel = int(eye_center_pixels[1])
                            target_pose = self.daemon_client.look_at_image(u_pixel, v_pixel)
                        except Exception as e:
                            # Timeout or API error - skip this frame, continue with previous tracking
                            logger.debug(f"IK call timeout/error, skipping frame: {e}")
                            continue

                        # Extract translation and rotation from the target pose directly
                        translation = target_pose[:3, 3]
                        rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

                        # Pitch: Use directly from daemon with offset (positive offset = look more down)
                        pitch = rotation[1] + self._pitch_offset

                        # Yaw: Use raw IK result directly (like original conversation app)
                        # The IK may return world-frame or body-relative depending on implementation
                        # Original app uses: yaw = -rotation[2] (inverted for mirrored camera)
                        # We don't invert because our camera isn't mirrored
                        yaw = rotation[2]

                        # Don't use world-frame target storage
                        with self._world_frame_target_lock:
                            self._world_frame_target_yaw = None

                        # Get current body yaw for debug display
                        if self.movement_manager:
                            body_yaw = self.movement_manager.get_current_body_yaw_rad()
                            self._last_body_yaw_deg = np.rad2deg(body_yaw)
                        else:
                            self._last_body_yaw_deg = 0.0

                        # Store debug visualization values
                        self._last_world_yaw_deg = 0.0  # Not calculated
                        self._last_yaw_deg = np.rad2deg(yaw)

                        # Update last sampled values for continuous offset updates
                        self._last_translation = translation
                        self._last_roll = rotation[0]

                        # Deadband: Only start new interpolation if change is significant enough
                        # This prevents detection jitter from causing rapid oscillations
                        pitch_change_deg = abs(np.rad2deg(pitch - self._current_interpolated_pitch))
                        yaw_change_deg = abs(np.rad2deg(yaw - self._current_interpolated_yaw))

                        min_change_threshold = 1.0  # degrees - ignore changes smaller than this

                        # Start pitch interpolation only if change is significant
                        if pitch_change_deg > min_change_threshold:
                            self._pitch_interpolation_target = pitch
                            self._pitch_interpolation_start = self._current_interpolated_pitch
                            self._pitch_interpolation_start_time = current_time

                        # Start yaw interpolation only if change is significant
                        if yaw_change_deg > min_change_threshold:
                            self._yaw_interpolation_target = yaw
                            self._yaw_interpolation_start = self._current_interpolated_yaw
                            self._yaw_interpolation_start_time = current_time

                        # Store face center for debug visualization
                        self._last_face_center = (int(eye_center_pixels[0]), int(eye_center_pixels[1]))

                    # No face detected while tracking enabled - set face lost timestamp
                    elif self.last_face_detected_time is None or self.last_face_detected_time == current_time:
                        # Only update if we haven't already set a face lost time
                        # (current_time check prevents overriding the disable-triggered timestamp)
                        pass

                    # Handle smooth interpolation (works for both face-lost and tracking-disabled cases)
                    if self.last_face_detected_time is not None:
                        time_since_face_lost = current_time - self.last_face_detected_time

                        if time_since_face_lost >= self.face_lost_delay:
                            # Start interpolation if not already started
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time

                                # Clear world-frame target (no longer tracking)
                                with self._world_frame_target_lock:
                                    self._world_frame_target_yaw = None
                                self._last_target_pixel_x = None  # Reset pixel tracking

                                # Capture current pose as start of interpolation
                                with self.face_tracking_lock:
                                    current_translation = self.face_tracking_offsets[:3]
                                    current_rotation_euler = self.face_tracking_offsets[3:]
                                    # Convert to 4x4 pose matrix
                                    pose_matrix = np.eye(4, dtype=np.float32)
                                    pose_matrix[:3, 3] = current_translation
                                    pose_matrix[:3, :3] = R.from_euler(
                                        "xyz", current_rotation_euler,
                                    ).as_matrix()
                                    self.interpolation_start_pose = pose_matrix

                            # Calculate interpolation progress (t from 0 to 1)
                            elapsed_interpolation = current_time - self.interpolation_start_time
                            t = min(1.0, elapsed_interpolation / self.interpolation_duration)

                            # Interpolate between current pose and neutral pose
                            interpolated_pose = linear_pose_interpolation(
                                self.interpolation_start_pose, neutral_pose, t,
                            )

                            # Extract translation and rotation from interpolated pose
                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

                            # Thread-safe update: interpolating back to zero offsets (neutral)
                            # Face tracking offsets should go to zero when face is lost
                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    0.0,  # x translation
                                    0.0,  # y translation
                                    0.0,  # z translation
                                    rotation[0],  # roll (interpolating to 0)
                                    rotation[1],  # pitch (interpolating to 0)
                                    rotation[2],  # yaw (interpolating to 0)
                                ]

                            # If interpolation is complete, reset timing and pitch state
                            if t >= 1.0:
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None
                                # Reset all tracking state to neutral
                                self._current_interpolated_pitch = np.deg2rad(0.0)
                                self._current_interpolated_yaw = np.deg2rad(0.0)
                                self._last_roll = 0.0  # Reset roll to prevent residual offset
                                self._pitch_interpolation_target = None
                                self._pitch_interpolation_start = None
                                self._pitch_interpolation_start_time = None
                                self._yaw_interpolation_target = None
                                self._yaw_interpolation_start = None
                                self._yaw_interpolation_start_time = None
                        # else: Keep current offsets (within 2s delay period)

                    # Debug visualization
                    if self.debug_window and frame is not None:
                        try:
                            debug_frame = frame.copy()
                            h, w = debug_frame.shape[:2]

                            # Draw center crosshair (image center reference)
                            center_x, center_y = w // 2, h // 2
                            cv2.line(debug_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
                            cv2.line(debug_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
                            cv2.putText(debug_frame, "CENTER", (center_x + 10, center_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                            # Determine face detection state and recovery action
                            face_detected_now = (eye_center is not None)
                            recovery_state = "TRACKING"
                            recovery_color = (0, 255, 0)  # Green
                            time_info = ""

                            if not face_detected_now:
                                if self.last_face_detected_time is None:
                                    recovery_state = "NEUTRAL"
                                    recovery_color = (128, 128, 128)  # Gray
                                else:
                                    time_since_lost = current_time - self.last_face_detected_time
                                    if time_since_lost < self.face_lost_delay:
                                        recovery_state = "GRACE PERIOD"
                                        recovery_color = (0, 255, 255)  # Yellow
                                        time_info = f"{time_since_lost:.1f}s / {self.face_lost_delay:.1f}s"
                                    elif self.interpolation_start_time is not None:
                                        recovery_state = "INTERPOLATING"
                                        recovery_color = (0, 165, 255)  # Orange
                                        elapsed = current_time - self.interpolation_start_time
                                        progress = min(1.0, elapsed / self.interpolation_duration) * 100
                                        time_info = f"{progress:.0f}%"
                                    else:
                                        recovery_state = "WAITING"
                                        recovery_color = (128, 128, 128)  # Gray

                            # Draw face center if detected
                            if self._last_face_center is not None and face_detected_now:
                                fx, fy = self._last_face_center
                                # Draw face center point
                                cv2.circle(debug_frame, (fx, fy), 10, (255, 0, 0), 2)
                                cv2.circle(debug_frame, (fx, fy), 3, (255, 0, 0), -1)

                                # Draw line from center to face
                                cv2.line(debug_frame, (center_x, center_y), (fx, fy), (255, 255, 0), 2)

                                # Display world yaw, body yaw, and relative yaw
                                cv2.putText(debug_frame, f"World yaw: {self._last_world_yaw_deg:+.1f} deg",
                                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                cv2.putText(debug_frame, f"Body yaw: {self._last_body_yaw_deg:+.1f} deg",
                                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                                cv2.putText(debug_frame, f"Relative yaw: {self._last_yaw_deg:+.1f} deg",
                                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                                # Direction based on relative yaw
                                if self._last_yaw_deg > 2:
                                    direction = "TURN RIGHT"
                                    color = (0, 165, 255)  # Orange
                                elif self._last_yaw_deg < -2:
                                    direction = "TURN LEFT"
                                    color = (255, 0, 255)  # Magenta
                                else:
                                    direction = "CENTERED"
                                    color = (0, 255, 0)  # Green

                                cv2.putText(debug_frame, direction, (20, 140),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

                                # Show face position offset from center
                                offset_x = fx - center_x
                                offset_y = fy - center_y
                                cv2.putText(debug_frame, f"Face offset: ({offset_x:+d}, {offset_y:+d}) px",
                                            (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

                            # Face detection status (always show)
                            face_status = f"Face: {'DETECTED' if face_detected_now else 'LOST'}"
                            face_color = (0, 255, 0) if face_detected_now else (0, 0, 255)
                            cv2.putText(debug_frame, face_status, (20, 220),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

                            # Recovery state (always show)
                            state_text = f"State: {recovery_state}"
                            if time_info:
                                state_text += f" ({time_info})"
                            cv2.putText(debug_frame, state_text, (20, 250),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, recovery_color, 2)

                            # Display tracking status
                            status = "TRACKING ON" if self.is_head_tracking_enabled else "TRACKING OFF"
                            status_color = (0, 255, 0) if self.is_head_tracking_enabled else (0, 0, 255)
                            cv2.putText(debug_frame, status, (w - 200, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                            # Show debug window
                            cv2.imshow("Face Tracking Debug", debug_frame)
                            cv2.waitKey(1)  # Process window events
                        except Exception as e:
                            # Silently disable debug window if display isn't available
                            logger.debug(f"Debug window disabled: {e}")
                            self.debug_window = False

                # Small sleep to prevent excessive CPU usage (same as main_works.py)
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)  # Longer sleep on error

        logger.debug("Camera worker thread exited")
