#!/usr/bin/env python3
"""
Side-by-side comparison test for YOLO vs MediaPipe face tracking.

This script runs both trackers on the same camera feed independently
(no daemon, no main script) to compare:
- Detection accuracy
- Inference speed
- Pose estimation quality
- CPU usage
"""

import time
import cv2
import numpy as np
from typing import Dict, List
import argparse

# Import daemon media wrapper for camera access
try:
    from daemon_media_wrapper import DaemonMediaWrapper
    DAEMON_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] DaemonMediaWrapper not available: {e}")
    DAEMON_AVAILABLE = False

# Import both trackers
try:
    from yolo_head_tracker import HeadTracker as YoloHeadTracker
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] YOLO tracker not available: {e}")
    YOLO_AVAILABLE = False

try:
    from mediapipe_head_tracker import HeadTracker as MediaPipeHeadTracker
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] MediaPipe tracker not available: {e}")
    MEDIAPIPE_AVAILABLE = False


class TrackerComparison:
    """Compares YOLO and MediaPipe head trackers side-by-side."""

    def __init__(self, use_daemon: bool = True, display_width: int = 640):
        """Initialize comparison test.

        Args:
            use_daemon: Use daemon's camera feed (True) or direct camera access (False)
            display_width: Width for each tracker display panel
        """
        self.use_daemon = use_daemon
        self.display_width = display_width
        self.cap = None
        self.media_manager = None

        # Initialize camera
        if use_daemon and DAEMON_AVAILABLE:
            print("[INIT] Connecting to daemon camera feed at http://localhost:8100...")
            try:
                self.media_manager = DaemonMediaWrapper("http://localhost:8100")
                print("[INIT] ✓ Daemon camera connected")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to daemon: {e}")
        else:
            print(f"[INIT] Opening camera 0 directly...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera 0")
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("[INIT] ✓ Camera opened")

        # Initialize trackers
        self.yolo_tracker = None
        self.mediapipe_tracker = None

        if YOLO_AVAILABLE:
            print("[INIT] Loading YOLO tracker...")
            try:
                self.yolo_tracker = YoloHeadTracker(device="cpu")
                print("[INIT] ✓ YOLO tracker loaded")
            except Exception as e:
                print(f"[INIT] ✗ YOLO tracker failed: {e}")

        if MEDIAPIPE_AVAILABLE:
            print("[INIT] Loading MediaPipe tracker...")
            try:
                self.mediapipe_tracker = MediaPipeHeadTracker(device="cpu")
                print("[INIT] ✓ MediaPipe tracker loaded")
            except Exception as e:
                print(f"[INIT] ✗ MediaPipe tracker failed: {e}")

        # Performance metrics
        self.metrics = {
            'yolo': {'times': [], 'detections': 0, 'frames': 0},
            'mediapipe': {'times': [], 'detections': 0, 'frames': 0}
        }

        # Create window
        cv2.namedWindow("Tracker Comparison", cv2.WINDOW_NORMAL)

    def draw_pose_overlay(self, frame: np.ndarray, tracker_name: str,
                         face_detected: bool, pose_data: Dict,
                         inference_time_ms: float) -> np.ndarray:
        """Draw tracking visualization on frame.

        Args:
            frame: Input frame
            tracker_name: "YOLO" or "MediaPipe"
            face_detected: Whether face was detected
            pose_data: Dictionary with pose information
            inference_time_ms: Inference time in milliseconds

        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Header bar
        header_color = (0, 200, 0) if face_detected else (0, 0, 200)
        cv2.rectangle(overlay, (0, 0), (w, 50), header_color, -1)

        # Title
        cv2.putText(overlay, f"{tracker_name}", (10, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        # Inference time
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0
        cv2.putText(overlay, f"{inference_time_ms:.1f}ms ({fps:.1f} FPS)",
                   (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if face_detected and pose_data:
            # Draw face center or landmarks
            if 'face_center_px' in pose_data:
                fx, fy = pose_data['face_center_px']
                cv2.circle(overlay, (int(fx), int(fy)), 8, (255, 0, 255), -1)
                cv2.circle(overlay, (int(fx), int(fy)), 12, (255, 255, 0), 2)

            # Draw center crosshair
            center_x, center_y = w // 2, h // 2
            cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
            cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)

            # Display pose angles
            y_offset = 70
            if 'roll_deg' in pose_data:
                cv2.putText(overlay, f"Roll:  {pose_data['roll_deg']:+6.1f} deg",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30

            if 'pitch_deg' in pose_data:
                cv2.putText(overlay, f"Pitch: {pose_data['pitch_deg']:+6.1f} deg",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30

            if 'yaw_deg' in pose_data:
                cv2.putText(overlay, f"Yaw:   {pose_data['yaw_deg']:+6.1f} deg",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Direction indicator
                yaw = pose_data['yaw_deg']
                if abs(yaw) < 5:
                    direction = "CENTERED"
                    color = (0, 255, 0)
                elif yaw > 0:
                    direction = f"LEFT {abs(yaw):.0f}°"
                    color = (0, 165, 255)
                else:
                    direction = f"RIGHT {abs(yaw):.0f}°"
                    color = (255, 0, 255)

                cv2.putText(overlay, direction, (10, h - 20),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        else:
            # No face detected
            cv2.putText(overlay, "NO FACE DETECTED", (10, h // 2),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        return overlay

    def test_yolo_tracker(self, frame: np.ndarray) -> tuple:
        """Test YOLO tracker on frame.

        Returns:
            (face_detected, pose_data, inference_time_ms)
        """
        if self.yolo_tracker is None:
            return False, {}, 0.0

        start_time = time.time()

        # YOLO returns (eye_center, roll) in [-1, 1] coordinates
        eye_center, roll = self.yolo_tracker.get_head_position(frame)

        inference_time_ms = (time.time() - start_time) * 1000.0

        face_detected = eye_center is not None

        pose_data = {}
        if face_detected:
            h, w = frame.shape[:2]
            # Convert [-1, 1] to pixel coordinates
            eye_center_norm = (eye_center + 1) / 2
            fx = eye_center_norm[0] * w
            fy = eye_center_norm[1] * h

            pose_data = {
                'face_center_px': (fx, fy),
                'roll_deg': np.rad2deg(roll) if roll else 0.0,
                # YOLO doesn't provide pitch/yaw directly
            }

        return face_detected, pose_data, inference_time_ms

    def test_mediapipe_tracker(self, frame: np.ndarray) -> tuple:
        """Test MediaPipe tracker on frame.

        Returns:
            (face_detected, pose_data, inference_time_ms)
        """
        if self.mediapipe_tracker is None:
            return False, {}, 0.0

        start_time = time.time()

        # MediaPipe returns (roll, pitch, yaw) in radians
        head_pose = self.mediapipe_tracker.get_head_pose(frame)

        inference_time_ms = (time.time() - start_time) * 1000.0

        face_detected = head_pose is not None

        pose_data = {}
        if face_detected:
            roll, pitch, yaw = head_pose

            # Also get face center for visualization
            eye_center, _ = self.mediapipe_tracker.get_head_position(frame)

            h, w = frame.shape[:2]
            if eye_center is not None:
                eye_center_norm = (eye_center + 1) / 2
                fx = eye_center_norm[0] * w
                fy = eye_center_norm[1] * h
                pose_data['face_center_px'] = (fx, fy)

            pose_data.update({
                'roll_deg': np.rad2deg(roll),
                'pitch_deg': np.rad2deg(pitch),
                'yaw_deg': np.rad2deg(yaw),
            })

        return face_detected, pose_data, inference_time_ms

    def run_comparison(self, duration_seconds: int = 60):
        """Run side-by-side comparison test.

        Args:
            duration_seconds: How long to run the test (0 = infinite)
        """
        print("\n" + "="*80)
        print("TRACKER COMPARISON TEST")
        print("="*80)
        print(f"YOLO:      {'✓ Ready' if self.yolo_tracker else '✗ Not Available'}")
        print(f"MediaPipe: {'✓ Ready' if self.mediapipe_tracker else '✗ Not Available'}")
        print("="*80)
        print("\nPress 'q' to quit, 's' to save screenshot")
        print()

        start_time = time.time()
        frame_count = 0

        try:
            while True:
                # Check duration
                if duration_seconds > 0 and (time.time() - start_time) > duration_seconds:
                    break

                # Capture frame
                if self.media_manager:
                    frame = self.media_manager.get_frame()
                    if frame is None:
                        print("[ERROR] Failed to capture frame from daemon")
                        break
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[ERROR] Failed to capture frame")
                        break

                frame_count += 1

                # Test both trackers on the same frame
                yolo_detected, yolo_pose, yolo_time = self.test_yolo_tracker(frame.copy())
                mp_detected, mp_pose, mp_time = self.test_mediapipe_tracker(frame.copy())

                # Update metrics
                if self.yolo_tracker:
                    self.metrics['yolo']['frames'] += 1
                    self.metrics['yolo']['times'].append(yolo_time)
                    if yolo_detected:
                        self.metrics['yolo']['detections'] += 1

                if self.mediapipe_tracker:
                    self.metrics['mediapipe']['frames'] += 1
                    self.metrics['mediapipe']['times'].append(mp_time)
                    if mp_detected:
                        self.metrics['mediapipe']['detections'] += 1

                # Create side-by-side display
                yolo_frame = self.draw_pose_overlay(frame.copy(), "YOLO",
                                                    yolo_detected, yolo_pose, yolo_time)
                mp_frame = self.draw_pose_overlay(frame.copy(), "MediaPipe",
                                                  mp_detected, mp_pose, mp_time)

                # Resize if needed
                if yolo_frame.shape[1] != self.display_width:
                    scale = self.display_width / yolo_frame.shape[1]
                    new_height = int(yolo_frame.shape[0] * scale)
                    yolo_frame = cv2.resize(yolo_frame, (self.display_width, new_height))
                    mp_frame = cv2.resize(mp_frame, (self.display_width, new_height))

                # Combine horizontally
                combined = np.hstack([yolo_frame, mp_frame])

                # Add comparison stats overlay
                stats_text = f"Frame {frame_count} | Runtime: {time.time() - start_time:.1f}s"
                cv2.putText(combined, stats_text, (10, combined.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Display
                cv2.imshow("Tracker Comparison", combined)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"tracker_comparison_{int(time.time())}.png"
                    cv2.imwrite(filename, combined)
                    print(f"[INFO] Screenshot saved: {filename}")

        except KeyboardInterrupt:
            print("\n[INFO] Test interrupted by user")

        finally:
            self.cleanup()
            self.print_summary()

    def print_summary(self):
        """Print comparison summary statistics."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        for name, metrics in self.metrics.items():
            if metrics['frames'] == 0:
                continue

            print(f"\n{name.upper()}:")
            print(f"  Frames processed:  {metrics['frames']}")
            print(f"  Faces detected:    {metrics['detections']} ({metrics['detections']/metrics['frames']*100:.1f}%)")

            if metrics['times']:
                avg_time = np.mean(metrics['times'])
                min_time = np.min(metrics['times'])
                max_time = np.max(metrics['times'])
                std_time = np.std(metrics['times'])

                print(f"  Inference time:")
                print(f"    Average: {avg_time:.1f} ms ({1000/avg_time:.1f} FPS)")
                print(f"    Min:     {min_time:.1f} ms ({1000/min_time:.1f} FPS)")
                print(f"    Max:     {max_time:.1f} ms ({1000/max_time:.1f} FPS)")
                print(f"    StdDev:  {std_time:.1f} ms")

        print("\n" + "="*80)

    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO vs MediaPipe face tracking")
    parser.add_argument("--no-daemon", action="store_true", help="Use direct camera access instead of daemon")
    parser.add_argument("--duration", type=int, default=0, help="Test duration in seconds (0=infinite)")
    parser.add_argument("--width", type=int, default=640, help="Display width per tracker")

    args = parser.parse_args()

    if not YOLO_AVAILABLE and not MEDIAPIPE_AVAILABLE:
        print("[ERROR] No trackers available. Please install dependencies:")
        print("  YOLO: pip install ultralytics supervision")
        print("  MediaPipe: pip install mediapipe")
        return

    try:
        comparison = TrackerComparison(
            use_daemon=not args.no_daemon,
            display_width=args.width
        )
        comparison.run_comparison(duration_seconds=args.duration)
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
