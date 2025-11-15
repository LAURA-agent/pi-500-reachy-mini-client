#!/usr/bin/env python3
"""MediaPipe-based head pose tracker using SolvePnP algorithm.

This tracker calculates 3D head orientation directly from facial landmarks
without requiring network calls or IK lookups.
"""

from __future__ import annotations
import logging
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class HeadTracker:
    """Calculates 3D head pose using MediaPipe Face Mesh.

    This tracker uses the Perspective-n-Point (PnP) algorithm to estimate
    the head's rotation in 3D space from a single 2D image.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize MediaPipe-based head tracker.

        Args:
            device: This argument is kept for compatibility but MediaPipe
                    manages its own delegates (CPU/GPU) automatically.
        """
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Face Mesh model loaded successfully.")
        except ImportError as e:
            logger.error("MediaPipe not installed. Install with: pip install mediapipe")
            raise ImportError(
                "MediaPipe required for this tracker. Install with: pip install mediapipe"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load MediaPipe model: {e}")
            raise

        # A generic 3D model of a human face corresponding to MediaPipe landmarks.
        # These points are in arbitrary units (e.g., millimeters).
        # Landmark indices: nose_tip=1, chin=152, left_eye_left=33, right_eye_right=263,
        #                   left_mouth=61, right_mouth=291
        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0],        # Nose tip (landmark 1)
            [0.0, -330.0, -65.0],    # Chin (landmark 152)
            [-225.0, 170.0, -135.0], # Left eye left corner (landmark 33)
            [225.0, 170.0, -135.0],  # Right eye right corner (landmark 263)
            [-150.0, -150.0, -125.0],# Left Mouth corner (landmark 61)
            [150.0, -150.0, -125.0], # Right mouth corner (landmark 291)
        ], dtype=np.float64)

    def get_head_pose(self, img: NDArray[np.uint8]) -> Tuple[float, float, float] | None:
        """Get head orientation (roll, pitch, yaw) from a single image.

        Args:
            img: Input image in BGR format.

        Returns:
            A tuple of (roll, pitch, yaw) in radians, or None if no face is detected.
        """
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False  # Performance optimization
        results = self.face_mesh.process(img_rgb)
        img_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # The 2D image points corresponding to our 3D model points.
        face_2d_image_points = np.array([
            [face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h],      # Nose
            [face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h],   # Chin
            [face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h],     # Left eye
            [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h],   # Right eye
            [face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h],     # Left mouth
            [face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h],   # Right mouth
        ], dtype=np.float64)

        # --- SolvePnP Estimation ---
        # A reasonable guess for the camera's intrinsic parameters.
        focal_length = w
        camera_center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Assuming no lens distortion.
        dist_coeffs = np.zeros((4, 1))

        # Solve the PnP problem to get the rotation and translation vectors.
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.face_3d_model_points,
            face_2d_image_points,
            camera_matrix,
            dist_coeffs,
        )

        if not success:
            return None

        # Convert the rotation vector to a rotation matrix, then to Euler angles.
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose = R.from_matrix(rotation_matrix)

        # Returns roll, pitch, yaw in RADIANS
        roll, pitch, yaw = pose.as_euler('xyz', degrees=False)

        # We negate pitch because a positive rotation around the x-axis in computer
        # vision corresponds to looking down, but for the robot, it's looking up.
        # We also adjust yaw to be more intuitive for tracking.
        return roll, -pitch, yaw

    def get_head_position(self, img: NDArray[np.uint8]) -> Tuple[NDArray[np.float32] | None, float | None]:
        """Get head position in YOLO-compatible format for comparison testing.

        This method exists for compatibility with the existing YOLO tracker interface.
        It converts MediaPipe's 3D pose to a 2D center point for visualization.

        Args:
            img: Input image in BGR format.

        Returns:
            Tuple of (eye_center [-1,1], roll_angle) for compatibility with YOLO interface.
        """
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.face_mesh.process(img_rgb)
        img_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]

        # Calculate face center from nose landmark
        nose_x = face_landmarks.landmark[1].x * w
        nose_y = face_landmarks.landmark[1].y * h

        # Normalize to [-1, 1] coordinates like YOLO tracker
        norm_x = (nose_x / w) * 2.0 - 1.0
        norm_y = (nose_y / h) * 2.0 - 1.0

        face_center = np.array([norm_x, norm_y], dtype=np.float32)

        # Get roll angle
        pose = self.get_head_pose(img)
        roll = pose[0] if pose else 0.0

        return face_center, roll
