from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np
from rtmlib import RTMPose

from app.config import batch_settings
from app.pose_service.draw_pose import draw_poses_numba


@dataclass
class ModelConfig:
    """Model configuration with validation"""

    height: int = 192
    width: int = 256

    def __post_init__(self):
        # Validate dimensions match the specific model requirements
        assert self.height == 192, f"RTMPose-M requires height=192, got {self.height}"
        assert self.width == 256, f"RTMPose-M requires width=256, got {self.width}"


class RTLib:
    """Simplified RTMPose wrapper focused on correctness over error handling"""

    def __init__(self, settings: batch_settings = batch_settings):
        self.settings = settings
        self.model_config = ModelConfig()
        self.pose_service = self._init_model()

    def _init_model(self) -> RTMPose:
        """Initialize model with basic validation"""
        if not self.settings.onnx_model:
            raise ValueError("ONNX model path required")

        return RTMPose(
            onnx_model=self.settings.onnx_model,
            backend=self.settings.backend,
            device=self.settings.device
        )

    def _resize_to_model_input(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resize image to exact model requirements"""
        h, w = image.shape[:2]

        # Calculate scale factors for coordinate mapping back to original size
        scale_x = w / self.model_config.width  # scale for x-coordinates
        scale_y = h / self.model_config.height  # scale for y-coordinates

        # CRITICAL: cv2.resize takes (width, height) tuple
        # Our model needs width=256, height=192
        opencv_size = (self.model_config.width, self.model_config.height)  # (256, 192)

        resized = cv2.resize(image, opencv_size, interpolation=cv2.INTER_LINEAR)

        # Verify the result matches model expectations
        expected_shape = (self.model_config.height, self.model_config.width, 3)  # (192, 256, 3)
        if resized.shape != expected_shape:
            raise RuntimeError(
                f"Resize operation failed. Expected {expected_shape}, got {resized.shape}. "
                f"Model requires height={self.model_config.height}, width={self.model_config.width}"
            )

        return resized, (scale_x, scale_y)

    def _scale_keypoints_to_original(self, keypoints: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
        """Map keypoints back to original image coordinates"""
        if keypoints is None:
            return None

        scaled = keypoints.copy()
        scale_x, scale_y = scale_factors

        scaled[:, :, 0] *= scale_x  # x coordinates
        scaled[:, :, 1] *= scale_y  # y coordinates

        return scaled

    async def predict(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run pose estimation and return original-size coordinates

        Returns:
            keypoints: Shape (N, K, 2) in original image coordinates
            scores: Shape (N, K) confidence scores
        """
        # Input validation - fail fast if image is invalid
        if image is None or image.size == 0:
            return None, None

        # Preprocess to exact model requirements
        model_input, scale_factors = self._resize_to_model_input(image)

        # Run inference - let any model errors propagate naturally
        keypoints, scores = self.pose_service(model_input)

        # Post-process coordinates back to original scale
        if keypoints is not None:
            keypoints = self._scale_keypoints_to_original(keypoints, scale_factors)

        return keypoints, scores

    async def predict_and_visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Complete pipeline: predict + visualize

        Returns:
            vis_image: Original image with pose overlay
        """
        keypoints, scores = await self.predict(image)

        if keypoints is None:
            return image

        # Use the optimized drawing function from draw_pose.py
        return draw_poses_numba(image, keypoints, scores)


# Global instance
rtlib_service = RTLib()
