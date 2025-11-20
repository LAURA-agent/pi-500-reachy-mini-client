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
        self.frame_endpoint = f"{self.daemon_url}/api/camera/frame"
        self._error_logged = False  # Only log error once to avoid spam
        logger.info(f"Initialized DaemonMediaWrapper for {self.daemon_url}")

    def get_frame(self) -> NDArray[np.uint8] | None:
        """Get latest camera frame from daemon.

        Returns:
            Frame as numpy array (H, W, 3) BGR format, or None if unavailable
        """
        try:
            response = requests.get(self.frame_endpoint, timeout=1.0)
            response.raise_for_status()

            # Decode JPEG bytes to numpy array
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                if not self._error_logged:
                    logger.warning("Failed to decode frame from daemon")
                    self._error_logged = True
                return None

            return frame

        except requests.Timeout:
            # Timeout is acceptable for high-frequency polling
            return None
        except requests.RequestException as e:
            # Only log the error once to avoid spam (e.g., 404 if endpoint doesn't exist)
            if not self._error_logged:
                logger.error(f"Failed to get frame from daemon: {e}")
                logger.info("Camera frame endpoint may not be available - this is normal in passive mode")
                self._error_logged = True
            return None
