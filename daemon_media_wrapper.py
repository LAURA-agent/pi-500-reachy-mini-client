"""Daemon Media Manager Wrapper.

Provides MediaManager-compatible interface that fetches frames from daemon API.
"""

import cv2
import numpy as np
import requests
import logging
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DaemonMediaWrapper:
    """Wrapper that mimics MediaManager but fetches frames from daemon API."""

    def __init__(self, daemon_url: str = "http://localhost:8100"):
        """Initialize wrapper.

        Args:
            daemon_url: Base URL of daemon API
        """
        self.daemon_url = daemon_url.rstrip("/")
        self.frame_endpoint = f"{self.daemon_url}/api/camera/stream.mjpg"
        logger.info(f"Initialized DaemonMediaWrapper for {self.daemon_url}")

    def get_frame(self) -> NDArray[np.uint8] | None:
        """Get latest camera frame from daemon.

        Returns:
            Frame as numpy array (H, W, 3) BGR format, or None if unavailable
        """
        try:
            # For MJPEG streams, we need to read a single frame from the multipart stream
            response = requests.get(self.frame_endpoint, timeout=1.0, stream=True)
            response.raise_for_status()

            # Read first frame from MJPEG stream
            # MJPEG format: --frame\r\nContent-Type: image/jpeg\r\n\r\n{JPEG bytes}\r\n
            bytes_buffer = b''
            for chunk in response.iter_content(chunk_size=1024):
                bytes_buffer += chunk

                # Look for JPEG start (FFD8) and end (FFD9) markers
                jpeg_start = bytes_buffer.find(b'\xff\xd8')
                jpeg_end = bytes_buffer.find(b'\xff\xd9', jpeg_start)

                if jpeg_start != -1 and jpeg_end != -1:
                    # Extract JPEG frame
                    jpeg_bytes = bytes_buffer[jpeg_start:jpeg_end + 2]
                    img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is None:
                        logger.warning("Failed to decode frame from daemon")
                        return None

                    return frame

            logger.warning("No complete JPEG frame found in stream")
            return None

        except requests.Timeout:
            # Timeout is acceptable for high-frequency polling
            return None
        except requests.RequestException as e:
            logger.error(f"Failed to get frame from daemon: {e}")
            return None
