# app/video_service/batch_GPU_process_onnx.py

import asyncio
from logging import getLogger
from typing import List, Tuple

import cv2
import numpy as np
from mmdet.apis import init_detector
from mmpose.utils import adapt_mmdet_pipeline

from app.config import batch_settings, settings
from app.pose_service.multi_person_onnx import MultiPersonONNXPoseEstimator

logger = getLogger(__name__)


class BatchPoseProcessorONNX:
    """Optimized batch processing with ONNX pose estimator with tracking support"""

    def __init__(self, config=None, onnx_file: str = None):
        self.config = config or batch_settings
        self.device = self.config.device or 'cuda:0'

        # Initialize detector first
        self._init_detector()

        # Initialize ONNX-based multi-person pose estimator with tracking
        self.onnx_file = onnx_file or self._get_default_onnx_path()
        self.pose_estimator = MultiPersonONNXPoseEstimator(self.onnx_file, self.device)
        self.pose_estimator.set_detector(self.detector)

        # Warm up models
        self._warm_up_models()
        logger.info("ONNX Batch processor with tracking initialized successfully")

    def _get_default_onnx_path(self) -> str:
        """Get default ONNX model path"""
        return "/home/stanley/jobs/python/AI/fastapi_mmpose/app/pose_service/configs/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504/end2end.onnx"

    def _init_detector(self):
        """初始化人体检测器"""
        self.detector = init_detector(
            settings.DETECTOR_CONFIG,
            settings.DETECTOR_CHECKPOINT,
            device=self.device
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        logger.info("Human detector initialized")

    def _warm_up_models(self):
        """Warm up models with dummy batch"""
        logger.info("Warming up ONNX models...")

        # Create dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run a few warmup inferences
        for _ in range(3):
            self.pose_estimator.inference(dummy_img)

        logger.info("ONNX model warmup completed")

    def reset_tracking(self):
        """重置tracking状态"""
        if hasattr(self.pose_estimator, 'reset_tracking'):
            self.pose_estimator.reset_tracking()
            logger.info("Tracking reset in batch processor")

    def set_tracking_enabled(self, enabled: bool):
        """启用/禁用tracking"""
        if hasattr(self.pose_estimator, 'set_tracking_enabled'):
            self.pose_estimator.set_tracking_enabled(enabled)
            logger.info(f"Tracking {'enabled' if enabled else 'disabled'} in batch processor")

    async def process_batch(self, batch_items: List[dict]) -> List[Tuple]:
        """Process batch of frames efficiently with ONNX pose estimator and tracking

        Args:
            batch_items: List of batch items with 'frame' key

        Returns:
            List of (batch_item, vis_img, pose_results) tuples
        """
        loop = asyncio.get_event_loop()

        def _process_frames():
            results = []

            for item in batch_items:
                frame = item['frame']

                # Ensure frame is in correct format
                processed_frame = self._ensure_frame_format(frame)

                # Process with multi-person ONNX estimator (now with tracking)
                vis_img, keypoints_list, scores_list = self.pose_estimator.inference(processed_frame)

                # Format results to match expected interface
                pose_results = self._format_pose_results(keypoints_list, scores_list)

                results.append((item, vis_img, pose_results))

            return results

        # Run processing in thread pool to avoid blocking
        return await loop.run_in_executor(None, _process_frames)

    def _ensure_frame_format(self, frame) -> np.ndarray:
        """Ensure frame is in correct format (BGR uint8)"""
        # Convert tensor to numpy if needed
        if hasattr(frame, 'cpu'):  # torch.Tensor
            frame = frame.cpu().numpy()

        # Ensure numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # Ensure uint8 type
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:  # Normalized to [0, 1]
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Ensure 3 channels
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return frame
        elif len(frame.shape) == 2:  # Grayscale
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            logger.warning(f"Unexpected frame shape: {frame.shape}")
            return frame

    def _format_pose_results(self, keypoints_list: List[np.ndarray],
                             scores_list: List[np.ndarray]) -> List[dict]:
        """Format pose results to match expected interface

        Args:
            keypoints_list: List of keypoints arrays for each detected person
            scores_list: List of score arrays for each detected person

        Returns:
            List of formatted pose results
        """
        if not keypoints_list:
            return []

        formatted_results = []

        for keypoints, scores in zip(keypoints_list, scores_list):
            # Create result structure that matches expected interface
            pose_result = PoseResultContainer(keypoints, scores)
            formatted_results.append(pose_result)

        return formatted_results


class PoseResultContainer:
    """Container to match the expected pose result interface"""

    def __init__(self, keypoints: np.ndarray, scores: np.ndarray):
        self.pred_instances = PredInstances(keypoints, scores)


class PredInstances:
    """Container for pose prediction instances"""

    def __init__(self, keypoints: np.ndarray, scores: np.ndarray):
        # Ensure correct shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[np.newaxis, ...]
        if len(scores.shape) == 1:
            scores = scores[np.newaxis, ...]

        self.keypoints = keypoints
        self.keypoint_scores = scores
