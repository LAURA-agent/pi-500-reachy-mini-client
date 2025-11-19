#!/usr/bin/env python3
"""Simple single-image tracker comparison test."""

import time
import cv2
import numpy as np

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


def test_on_image(image_path: str):
    """Test both trackers on a single image."""

    # Load image
    print(f"[INFO] Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    print(f"[INFO] Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Initialize trackers
    yolo_tracker = None
    mediapipe_tracker = None

    if YOLO_AVAILABLE:
        print("[INFO] Loading YOLO tracker...")
        yolo_tracker = YoloHeadTracker(device="cpu")
        print("[INFO] ✓ YOLO loaded")

    if MEDIAPIPE_AVAILABLE:
        print("[INFO] Loading MediaPipe tracker...")
        mediapipe_tracker = MediaPipeHeadTracker(device="cpu")
        print("[INFO] ✓ MediaPipe loaded")

    print("\n" + "="*80)
    print("TESTING YOLO TRACKER")
    print("="*80)

    if yolo_tracker:
        start = time.time()
        eye_center, roll = yolo_tracker.get_head_position(frame)
        elapsed_ms = (time.time() - start) * 1000

        print(f"Inference time: {elapsed_ms:.1f} ms ({1000/elapsed_ms:.1f} FPS)")
        if eye_center is not None:
            print(f"✓ Face detected")
            print(f"  Eye center (normalized): {eye_center}")
            print(f"  Roll angle: {np.rad2deg(roll):.1f}°" if roll else "  Roll: N/A")
        else:
            print("✗ No face detected")

    print("\n" + "="*80)
    print("TESTING MEDIAPIPE TRACKER")
    print("="*80)

    if mediapipe_tracker:
        start = time.time()
        head_pose = mediapipe_tracker.get_head_pose(frame)
        elapsed_ms = (time.time() - start) * 1000

        print(f"Inference time: {elapsed_ms:.1f} ms ({1000/elapsed_ms:.1f} FPS)")
        if head_pose is not None:
            roll, pitch, yaw = head_pose
            print(f"✓ Face detected")
            print(f"  Roll:  {np.rad2deg(roll):+6.1f}°")
            print(f"  Pitch: {np.rad2deg(pitch):+6.1f}°")
            print(f"  Yaw:   {np.rad2deg(yaw):+6.1f}°")

            # Interpret yaw
            if abs(np.rad2deg(yaw)) < 5:
                direction = "Looking straight ahead"
            elif yaw > 0:
                direction = f"Looking LEFT ({abs(np.rad2deg(yaw)):.0f}°)"
            else:
                direction = f"Looking RIGHT ({abs(np.rad2deg(yaw)):.0f}°)"
            print(f"  → {direction}")
        else:
            print("✗ No face detected")

    print("\n" + "="*80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 test_tracker_simple.py <image_path>")
        print("")
        print("Example:")
        print("  python3 test_tracker_simple.py face_photo.jpg")
        print("")
        print("Take a photo with:")
        print("  raspistill -o face_photo.jpg")
        sys.exit(1)

    test_on_image(sys.argv[1])
