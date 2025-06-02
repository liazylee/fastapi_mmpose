# import time  # noqa
#
# import numpy as np
# from mmdet.apis import init_detector, inference_detector
# from mmpose.apis import (inference_topdown, init_model as init_pose_estimator)
# from mmpose.evaluation.functional import nms
# from mmpose.registry import VISUALIZERS
# from mmpose.structures import merge_data_samples  # noqa
# from mmpose.utils import adapt_mmdet_pipeline
#
# from app.config import settings
# from app.pose_service.draw_pose import draw_poses_numba, draw_poses_gpu
# from app.video_service.helper import timeit
#
#
# class PoseService:
#     def __init__(self):
#         self.device = settings.DEVICE
#         self._init_models()
#         self._init_visualizer()
#         self.skeleton = self.pose_estimator.dataset_meta.get('skeleton_info', {}).get('skeleton', None)
#
#     def _init_models(self):
#         """Initialize detection and pose estimation models."""
#         print(settings.DETECTOR_CONFIG)
#         self.detector = init_detector(
#             settings.DETECTOR_CONFIG,
#             settings.DETECTOR_CHECKPOINT,
#             device=self.device
#         )
#         self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
#
#         self.pose_estimator = init_pose_estimator(
#             settings.POSE_CONFIG,
#             settings.POSE_CHECKPOINT,
#             device=self.device,
#             cfg_options=dict(
#                 model=dict(test_cfg=dict(output_heatmaps=True)),
#             )
#         )
#
#     def _init_visualizer(self):
#         """Initialize the visualizer for pose visualization."""
#         # 使用默认的可视化器配置
#         visualizer_cfg = dict(
#             type='PoseLocalVisualizer',
#             vis_backends=[dict(type='LocalVisBackend')],
#             name='visualizer',
#             radius=getattr(settings, 'RADIUS', 5),
#             alpha=getattr(settings, 'ALPHA', 0.8),
#             line_width=getattr(settings, 'LINE_WIDTH', 2),
#         )
#
#         self.visualizer = VISUALIZERS.build(visualizer_cfg)
#         self.visualizer.set_dataset_meta(
#             self.pose_estimator.dataset_meta,
#             skeleton_style='mmpose'
#         )
#
#     @timeit
#     def process_image(self, image):
#         """
#         Process a single image through detection and pose estimation pipeline.
#
#         Args:
#             image: Input image in BGR format
#
#         Returns:
#             tuple: (visualization image, pose results)
#         """
#         # 1. Person detection
#
#         det_result = inference_detector(self.detector, image)
#
#         pred_instance = det_result.pred_instances.cpu().numpy()
#
#         if pred_instance is None or len(pred_instance) == 0:
#             return image, None
#
#         # Filter detections based on category ID and score threshold
#         bboxes = np.concatenate(
#             (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
#         bboxes = bboxes[np.logical_and(pred_instance.labels == settings.det_cat_id,
#                                        pred_instance.scores > settings.bbox_thr)]
#         bboxes = bboxes[nms(bboxes, settings.nms_thr), :4]
#
#         if len(bboxes) == 0:
#             return image, None
#
#         # 2. Pose estimation
#
#         pose_results = inference_topdown(self.pose_estimator, image, bboxes)
#
#         if settings.DRAW_POSE_GPU:
#             # Use Numba for GPU-accelerated drawing
#             vis_img = draw_poses_gpu(image, pose_results, self.skeleton, self.device)
#         else:
#             vis_img = draw_poses_numba(image, pose_results, self.skeleton)
#         return vis_img, pose_results
#         # Create a singleton instance
#
#
# pose_service = PoseService()
