#!/usr/bin/env python3
"""Benchmark individual trackers separately to avoid CPU contention."""

import time
import numpy as np
import sys

# Import daemon wrapper
from daemon_media_wrapper import DaemonMediaWrapper


def benchmark_tracker(tracker_name: str, tracker, media_manager, duration: int = 20):
    """Benchmark a single tracker for specified duration.

    Args:
        tracker_name: Name for display
        tracker: Tracker instance with get_head_position() method
        media_manager: DaemonMediaWrapper instance
        duration: Test duration in seconds
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {tracker_name}")
    print(f"Duration: {duration} seconds")
    print(f"{'='*80}\n")

    times = []
    detections = 0
    frames = 0
    start_time = time.time()

    print("Running", end="", flush=True)

    try:
        while (time.time() - start_time) < duration:
            # Get frame
            frame = media_manager.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frames += 1

            # Time the detection
            detect_start = time.time()
            eye_center, _ = tracker.get_head_position(frame)
            detect_time = (time.time() - detect_start) * 1000.0

            times.append(detect_time)

            if eye_center is not None:
                detections += 1

            # Progress indicator every 50 frames
            if frames % 50 == 0:
                print(".", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    elapsed = time.time() - start_time
    print(f"\n\nTest completed in {elapsed:.1f} seconds\n")

    # Calculate statistics
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        print(f"{'='*80}")
        print(f"RESULTS: {tracker_name}")
        print(f"{'='*80}")
        print(f"Frames processed:    {frames}")
        print(f"Faces detected:      {detections} ({detections/frames*100:.1f}%)")
        print(f"\nInference Time:")
        print(f"  Average: {avg_time:.1f} ms  ({1000/avg_time:.1f} FPS)")
        print(f"  Min:     {min_time:.1f} ms  ({1000/min_time:.1f} FPS)")
        print(f"  Max:     {max_time:.1f} ms  ({1000/max_time:.1f} FPS)")
        print(f"  StdDev:  {std_time:.1f} ms")
        print(f"\nOverall:")
        print(f"  Total time:   {elapsed:.1f} seconds")
        print(f"  System FPS:   {frames/elapsed:.1f} FPS")
        print(f"{'='*80}\n")

        return {
            'frames': frames,
            'detections': detections,
            'avg_ms': avg_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'std_ms': std_time,
            'system_fps': frames/elapsed,
            'detection_rate': detections/frames*100,
        }
    else:
        print("[ERROR] No frames processed")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark face trackers individually")
    parser.add_argument("--tracker", choices=["yolo", "mediapipe", "both"], default="both",
                       help="Which tracker to benchmark")
    parser.add_argument("--duration", type=int, default=20,
                       help="Test duration in seconds per tracker")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("FACE TRACKER BENCHMARK")
    print("="*80)
    print(f"Test duration: {args.duration} seconds per tracker")
    print(f"Mode: {args.tracker}")
    print("="*80)

    # Connect to daemon
    print("\n[INIT] Connecting to daemon camera feed...")
    try:
        media_manager = DaemonMediaWrapper("http://localhost:8100")
        print("[INIT] ✓ Daemon camera connected")
    except Exception as e:
        print(f"[ERROR] Failed to connect to daemon: {e}")
        return

    results = {}

    # Test YOLO
    if args.tracker in ["yolo", "both"]:
        try:
            print("\n[INIT] Loading YOLO tracker...")
            from yolo_head_tracker import HeadTracker as YoloTracker
            yolo_tracker = YoloTracker(device="cpu")
            print("[INIT] ✓ YOLO tracker loaded")

            input("\nPress ENTER to start YOLO benchmark...")
            results['yolo'] = benchmark_tracker("YOLO", yolo_tracker, media_manager, args.duration)

        except Exception as e:
            print(f"[ERROR] YOLO test failed: {e}")
            import traceback
            traceback.print_exc()

    # Test MediaPipe FaceDetector
    if args.tracker in ["mediapipe", "both"]:
        try:
            print("\n[INIT] Loading MediaPipe FaceDetector (BlazeFace)...")
            from mediapipe_face_detector import HeadTracker as MediaPipeTracker
            mp_tracker = MediaPipeTracker(device="cpu")
            print("[INIT] ✓ MediaPipe FaceDetector loaded")

            if args.tracker == "both":
                input("\nPress ENTER to start MediaPipe benchmark...")
            else:
                input("\nPress ENTER to start benchmark...")

            results['mediapipe'] = benchmark_tracker("MediaPipe FaceDetector", mp_tracker, media_manager, args.duration)

        except Exception as e:
            print(f"[ERROR] MediaPipe test failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    if len(results) == 2:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        y = results['yolo']
        m = results['mediapipe']

        print(f"\n{'Metric':<25} {'YOLO':>15} {'MediaPipe':>15} {'Winner':>15}")
        print("-"*80)
        print(f"{'Detection Rate':<25} {y['detection_rate']:>14.1f}% {m['detection_rate']:>14.1f}% ", end="")
        print("YOLO" if y['detection_rate'] > m['detection_rate'] else "MediaPipe")

        print(f"{'Avg Inference (ms)':<25} {y['avg_ms']:>15.1f} {m['avg_ms']:>15.1f} ", end="")
        print("MediaPipe" if m['avg_ms'] < y['avg_ms'] else "YOLO")

        print(f"{'Theoretical FPS':<25} {1000/y['avg_ms']:>15.1f} {1000/m['avg_ms']:>15.1f} ", end="")
        print("MediaPipe" if m['avg_ms'] < y['avg_ms'] else "YOLO")

        print(f"{'System FPS':<25} {y['system_fps']:>15.1f} {m['system_fps']:>15.1f} ", end="")
        print("MediaPipe" if m['system_fps'] > y['system_fps'] else "YOLO")

        print(f"{'Consistency (StdDev)':<25} {y['std_ms']:>15.1f} {m['std_ms']:>15.1f} ", end="")
        print("MediaPipe" if m['std_ms'] < y['std_ms'] else "YOLO")

        # Speed advantage
        speedup = y['avg_ms'] / m['avg_ms']
        print(f"\nMediaPipe is {speedup:.1f}x faster than YOLO")

        print("="*80)


if __name__ == "__main__":
    main()
