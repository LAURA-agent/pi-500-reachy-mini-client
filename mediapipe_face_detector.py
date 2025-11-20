#!/usr/bin/env python3
"""MediaPipe FaceDetector - simpler, faster face detection without 3D pose."""

from __future__ import annotations
import logging
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class HeadTracker:
    """Simple face detection using MediaPipe FaceDetector (BlazeFace).

    This is the simpler model that just finds face locations,
    not the complex FaceMesh with 468 landmarks.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize MediaPipe FaceDetector."""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.5,
                model_selection=0,  # 0 = within 2 meters, 1 = within 5 meters
            )
            logger.info("MediaPipe FaceDetector loaded successfully.")
        except ImportError as e:
            logger.error("MediaPipe not installed. Install with: pip install mediapipe")
            raise ImportError(
                "MediaPipe required for this tracker. Install with: pip install mediapipe"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load MediaPipe model: {e}")
            raise

    def get_head_position(self, img: NDArray[np.uint8]) -> Tuple[NDArray[np.float32] | None, float | None]:
        """Get head position for compatibility with YOLO interface.

        Args:
            img: Input image in BGR format.

        Returns:
            Tuple of (face_center [-1,1], None) or (None, None) if no face detected.
        """
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.face_detection.process(img_rgb)
        img_rgb.flags.writeable = True

        if not results.detections:
            return None, None

        # Get first detection
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        # Calculate center of bounding box
        center_x = bboxC.xmin + bboxC.width / 2
        center_y = bboxC.ymin + bboxC.height / 2

        face_center = np.array([center_x, center_y], dtype=np.float32)

        return face_center, None  # No roll angle from FaceDetector
