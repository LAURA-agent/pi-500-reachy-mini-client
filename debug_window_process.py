"""Separate process for OpenCV debug window (macOS requirement).

macOS requires cv2.imshow to run on the main thread of a process.
This module runs in a separate process to display face tracking debug frames.
"""
import cv2
import numpy as np
from multiprocessing import Queue
import logging
import signal
import sys

logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("[DEBUG WINDOW] Received interrupt signal, shutting down...")
    cv2.destroyAllWindows()
    sys.exit(0)


def run_debug_window(frame_queue: Queue) -> None:
    """Run debug window in separate process (main thread).

    Args:
        frame_queue: Multiprocessing queue to receive frames from camera worker
    """
    # Set up signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[DEBUG WINDOW] Process started")

    # Create window on this process's main thread
    window_name = "Face Tracking Debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 100, 100)

    print("[DEBUG WINDOW] Window created and positioned")

    try:
        while True:
            # Get frame from queue (non-blocking with timeout)
            try:
                frame = frame_queue.get(timeout=0.1)

                if frame is None:
                    # None signals shutdown
                    print("[DEBUG WINDOW] Shutdown signal received")
                    break

                # Display frame
                cv2.imshow(window_name, frame)

                # Process window events (required for macOS)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("[DEBUG WINDOW] ESC pressed, closing")
                    break

            except Exception as e:
                # Queue timeout or error - just continue
                cv2.waitKey(1)  # Still need to process events
                continue

    finally:
        cv2.destroyAllWindows()
        print("[DEBUG WINDOW] Process terminated")
