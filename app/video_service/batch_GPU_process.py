import asyncio
from logging import getLogger

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import (inference_topdown, init_model as init_pose_estimator)
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from app.config import batch_settings, settings
from app.pose_service.draw_pose import draw_poses_numba
from app.video_service.helper import inference_topdown_batch, batch_inference_detector

logger = getLogger(__name__)


class BatchPoseProcessor:
    """Optimized batch processing """

    def __init__(self, config: batch_settings = batch_settings):
        self.config = config
        self.device = config.device or 'cuda:0'

        # Initialize models with batch optimization
        self._init_models()

        # GPU memory management
        self.memory_pool = self._create_memory_pool()

    def _create_memory_pool(self):
        """Create pre-allocated memory pool for efficient processing"""
        pool = {}

        try:
            # Pre-allocate common tensor sizes
            batch_size = self.config.batch_size
            h, w = self.config.input_resolution

            pool['input_tensor'] = torch.zeros(
                (batch_size, h, w, 3),
                dtype=torch.uint8,
                device=self.device,
                pin_memory=self.config.pin_memory
            )

            pool['normalized_tensor'] = torch.zeros(
                (batch_size, 3, h, w),
                dtype=torch.float32,
                device=self.device
            )

            logger.info(f"Memory pool created with {len(pool)} tensors")

        except Exception as e:
            logger.warning(f"Memory pool creation failed: {e}")
            pool = {}

        return pool

    def _warm_up_models(self):
        """Warm up models with dummy batch to initialize CUDA kernels"""
        logger.info("Warming up models...")

        # Create dummy batch
        dummy_size = self.config.input_resolution
        dummy_batch = np.random.randint(
            0, 255,
            (self.config.batch_size, dummy_size[1], dummy_size[0], 3),
            dtype=np.uint8
        )

        try:
            # Warmup detection
            for i in range(2):  # Run a few times to stabilize
                dummy_frame = dummy_batch[0]
                _ = inference_detector(self.detector, dummy_frame)

            # Warmup pose estimation with dummy detections
            dummy_boxes = np.array([[100, 100, 200, 300, 0.9]])  # x1,y1,x2,y2,score
            for i in range(2):
                _ = inference_topdown(self.pose_estimator, dummy_frame, dummy_boxes)

            logger.info("Model warmup completed")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _init_models(self):
        """Initialize models with batch-friendly settings"""
        # Detection model
        self.detector = init_detector(
            settings.DETECTOR_CONFIG,
            settings.DETECTOR_CHECKPOINT,
            device=self.device
        )

        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # Pose model
        self.pose_estimator = init_pose_estimator(
            settings.POSE_CONFIG,
            settings.POSE_CHECKPOINT,
            device=self.device,
            cfg_options=dict(
                model=dict(
                    test_cfg=dict(
                        output_heatmaps=False,  # Disable for speed
                        flip_test=False  # Disable flip augmentation
                    )
                )
            )
        )

        # Warm up models with dummy batch
        self._warm_up_models()
        logger.info("Models initialized successfully")

    async def process_batch(self, batch_items):
        """Process batch of frames efficiently"""
        try:
            # 1. Extract and prepare frames
            frames = [item['frame'] for item in batch_items]
            processed_frames = self._prepare_batch_tensor(frames)

            # 2. Batch detection (pass numpy arrays)
            detection_results = await self._batch_detect(processed_frames)

            # 3. Batch pose estimation
            pose_results = await self._batch_pose_estimation(
                processed_frames, detection_results
            )

            # 4. Batch visualization
            vis_results = await self._batch_visualize(processed_frames, pose_results)

            # 5. Return results
            return list(zip(batch_items, vis_results, pose_results))

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return original frames as fallback
            return [(item, item['frame'], None) for item in batch_items]

    async def _batch_visualize(self, frames, pose_results):
        """Batch visualization of pose results"""
        loop = asyncio.get_event_loop()

        def _visualize():
            vis_frames = []

            for frame, poses in zip(frames, pose_results):
                if poses and len(poses) > 0:
                    # Use the manual drawing method from original code
                    vis_frame = draw_poses_numba(
                        frame, poses,
                        skeleton=None,

                    )
                    # vis_frame = self._draw_poses_manually(frame, poses)
                else:
                    vis_frame = frame.copy()

                vis_frames.append(vis_frame)

            return vis_frames

        return await loop.run_in_executor(None, _visualize)

    def _draw_poses_manually(self, image, pose_results):
        """Manual pose drawing (copied from original code)"""
        vis_img = image.copy()

        # Use the same keypoint colors and skeleton from original code
        keypoint_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
        ]

        default_skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
        ]

        for pose_result in pose_results:
            if hasattr(pose_result, 'pred_instances'):
                pred_instances = pose_result.pred_instances

                if hasattr(pred_instances, 'keypoints') and hasattr(pred_instances, 'keypoint_scores'):
                    keypoints_array = pred_instances.keypoints
                    scores_array = pred_instances.keypoint_scores

                    num_persons = keypoints_array.shape[0] if len(keypoints_array.shape) > 2 else 1

                    for person_idx in range(num_persons):
                        if len(keypoints_array.shape) > 2:
                            keypoints = keypoints_array[person_idx]
                            scores = scores_array[person_idx]
                        else:
                            keypoints = keypoints_array
                            scores = scores_array

                        # Draw keypoints
                        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                            if score > 0.3:
                                x, y = int(kpt[0]), int(kpt[1])
                                if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                                    color = keypoint_colors[i % len(keypoint_colors)]
                                    cv2.circle(vis_img, (x, y), 2, color, -1)
                                    cv2.circle(vis_img, (x, y), 3, (255, 255, 255), 1)

                        # Draw skeleton
                        for connection in default_skeleton:
                            pt1_idx, pt2_idx = connection
                            if (pt1_idx < len(scores) and pt2_idx < len(scores) and
                                    scores[pt1_idx] > 0.3 and scores[pt2_idx] > 0.3):
                                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                                if (0 <= pt1[0] < vis_img.shape[1] and 0 <= pt1[1] < vis_img.shape[0] and
                                        0 <= pt2[0] < vis_img.shape[1] and 0 <= pt2[1] < vis_img.shape[0]):
                                    cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)

        return vis_img

    def _prepare_batch_tensor(self, frames) -> list:
        """Efficiently prepare batch tensor"""
        # Resize and normalize frames in batch
        processed_frames = []
        target_size = self.config.input_resolution

        for frame in frames:
            # Ensure numpy array
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()

            # Resize if needed
            if frame.shape[:2] != target_size[::-1]:  # OpenCV uses (width, height)
                frame = cv2.resize(frame, target_size)

            # Ensure uint8 format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            processed_frames.append(frame)

        return processed_frames  # Return list of numpy arrays, not tensor

    async def _batch_detect(self, batch_tensor)->list[np.ndarray]:
        """Batch person detection"""
        loop = asyncio.get_event_loop()

        def _detect():
            #results = []

            # Process each frame individually to avoid pipeline issues

            return batch_inference_detector(self.detector, batch_tensor)
            # for frame in batch_tensor:
            #     try:
            #         # Convert tensor back to numpy if needed
            #         if isinstance(frame, torch.Tensor):
            #             frame_np = frame.cpu().numpy()
            #         else:
            #             frame_np = frame
            #
            #         # Ensure correct format (BGR)
            #         if frame_np.dtype != np.uint8:
            #             frame_np = frame_np.astype(np.uint8)
            #
            #         # Run detection
            #         det_result = inference_detector(self.detector, frame_np)
            #         filtered_boxes = self._filter_detections(det_result)
            #         results.append(filtered_boxes)
            #
            #     except Exception as e:
            #         logger.error(f"Detection failed for frame: {e}")
            #         # Return empty detection for failed frame
            #         results.append(np.array([]).reshape(0, 5))

            # return results

        return await loop.run_in_executor(None, _detect)

    # def _filter_detections(self, det_result)->np.ndarray:
    #     """Filter and process detection results"""
    #     pred_instance = det_result.pred_instances.cpu().numpy()  # Ensure numpy array
    #
    #     if pred_instance is None or len(pred_instance) == 0:
    #         return np.array([]).reshape(0, 5)
    #
    #     # Filter by category (person) and confidence
    #     person_mask = np.logical_and(
    #         pred_instance.labels == settings.det_cat_id,
    #         pred_instance.scores > settings.bbox_thr
    #     )
    #
    #     if not person_mask.any():
    #         return np.array([]).reshape(0, 5)
    #
    #     # Get filtered boxes and scores
    #     boxes = pred_instance.bboxes[person_mask]
    #     scores = pred_instance.scores[person_mask]
    #
    #     # Combine boxes with scores
    #     detections = np.concatenate([boxes, scores[:, None]], axis=1)
    #
    #     # Apply NMS
    #
    #     keep_indices = nms(detections, settings.nms_thr)
    #
    #     return detections[keep_indices, :4]  # Return only box coordinates

    async def _batch_pose_estimation(self, frames, detection_results):
        """Batch pose estimation with smart batching"""
        loop = asyncio.get_event_loop()

        def _estimate_poses():

            try:
                # Process each frame with its detections
                pose_result = inference_topdown_batch(self.pose_estimator, frames, detection_results)
                return pose_result

            except Exception as e:
                logger.error(f"Pose estimation failed: {e}")
                # Return empty results for all frames
                all_pose_results = [[] for _ in frames]

            return all_pose_results

        return await loop.run_in_executor(None, _estimate_poses)

