import time  # noqa

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import (inference_topdown, init_model as init_pose_estimator)
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples  # noqa
from mmpose.utils import adapt_mmdet_pipeline

from app.config import settings
from app.video_service.helper import timeit

keypoint_colors = [
    (255, 0, 0),  # 0: nose
    (255, 85, 0),  # 1: left_eye
    (255, 170, 0),  # 2: right_eye
    (255, 255, 0),  # 3: left_ear
    (170, 255, 0),  # 4: right_ear
    (85, 255, 0),  # 5: left_shoulder
    (0, 255, 0),  # 6: right_shoulder
    (0, 255, 85),  # 7: left_elbow
    (0, 255, 170),  # 8: right_elbow
    (0, 255, 255),  # 9: left_wrist
    (0, 170, 255),  # 10: right_wrist
    (0, 85, 255),  # 11: left_hip
    (0, 0, 255),  # 12: right_hip
    (85, 0, 255),  # 13: left_knee
    (170, 0, 255),  # 14: right_knee
    (255, 0, 255),  # 15: left_ankle
    (255, 0, 170),  # 16: right_ankle
]
default_skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 躯干
    [5, 11], [6, 12], [5, 6],  # 肩膀连接
    [5, 7], [6, 8], [7, 9], [8, 10],  # 手臂
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
    [0, 5], [0, 6]  # 头部到肩膀 (可选)
]


class PoseService:
    def __init__(self):
        self.device = settings.DEVICE
        self._init_models()
        self._init_visualizer()

    def _init_models(self):
        """Initialize detection and pose estimation models."""
        print(settings.DETECTOR_CONFIG)
        self.detector = init_detector(
            settings.DETECTOR_CONFIG,
            settings.DETECTOR_CHECKPOINT,
            device=self.device
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        self.pose_estimator = init_pose_estimator(
            settings.POSE_CONFIG,
            settings.POSE_CHECKPOINT,
            device=self.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=True)),
            )
        )

    def _init_visualizer(self):
        """Initialize the visualizer for pose visualization."""
        # 使用默认的可视化器配置
        visualizer_cfg = dict(
            type='PoseLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend')],
            name='visualizer',
            radius=getattr(settings, 'RADIUS', 5),
            alpha=getattr(settings, 'ALPHA', 0.8),
            line_width=getattr(settings, 'LINE_WIDTH', 2),
        )

        self.visualizer = VISUALIZERS.build(visualizer_cfg)
        self.visualizer.set_dataset_meta(
            self.pose_estimator.dataset_meta,
            skeleton_style='mmpose'
        )

    @timeit
    def _draw_poses_manually(self, image, pose_results):
        """手动绘制关键点，作为备用方案"""
        vis_img = image.copy()

        # COCO 17关键点的骨架连接 (如果获取不到dataset_meta中的skeleton)
        # 这是标准的COCO-17关键点连接关系

        # 尝试获取骨架连接信息
        skeleton = self.pose_estimator.dataset_meta.get('skeleton_info', {}).get('skeleton', default_skeleton)
        if not skeleton:
            skeleton = default_skeleton

        # 定义关键点颜色 (BGR格式)

        for pose_idx, pose_result in enumerate(pose_results):

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

                        # 绘制关键点
                        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                            if score > 0.3:  # 置信度阈值
                                x, y = int(kpt[0]), int(kpt[1])

                                if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                                    color = keypoint_colors[i % len(keypoint_colors)]
                                    cv2.circle(vis_img, (x, y), 2, color, -1)
                                    cv2.circle(vis_img, (x, y), 3, (255, 255, 255), 1)  # 白色边框

                        connections_drawn = 0
                        for connection in skeleton:
                            pt1_idx, pt2_idx = connection

                            if (pt1_idx < len(scores) and pt2_idx < len(scores) and
                                    scores[pt1_idx] > 0.3 and scores[pt2_idx] > 0.3):

                                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))

                                if (0 <= pt1[0] < vis_img.shape[1] and 0 <= pt1[1] < vis_img.shape[0] and
                                        0 <= pt2[0] < vis_img.shape[1] and 0 <= pt2[1] < vis_img.shape[0]):
                                    cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)  # 绿色骨架线
                                    connections_drawn += 1
        return vis_img

    @timeit
    def process_image(self, image):
        """
        Process a single image through detection and pose estimation pipeline.

        Args:
            image: Input image in BGR format

        Returns:
            tuple: (visualization image, pose results)
        """
        # 1. Person detection

        det_result = inference_detector(self.detector, image)

        pred_instance = det_result.pred_instances.cpu().numpy()

        if pred_instance is None or len(pred_instance) == 0:
            return image, None

        # Filter detections based on category ID and score threshold
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == settings.det_cat_id,
                                       pred_instance.scores > settings.bbox_thr)]
        bboxes = bboxes[nms(bboxes, settings.nms_thr), :4]

        if len(bboxes) == 0:
            return image, None

        # 2. Pose estimation

        pose_results = inference_topdown(self.pose_estimator, image, bboxes)

        # data_samples = merge_data_samples(pose_results)
        # Method 1: Use add_datasample

        # start_time = time.time()
        # for pose_result in pose_results:
        #     self.visualizer.add_datasample(
        #         'result',
        #         image,
        #         data_sample=pose_result,
        #         draw_gt=False,
        #         draw_bbox=True,
        #         draw_heatmap=False,
        #         skeleton_style='mmpose',
        #         wait_time=0,
        #         kpt_thr=0.3,
        #         show=False,
        #     )

        # self.visualizer.add_datasample(
        #     'result',
        #     image,
        #     data_sample=pose_results,
        #     draw_gt=False,
        #     draw_heatmap=False,  # Do not draw heatmap
        #     draw_bbox=True,
        #     show_kpt_idx=True,
        #     skeleton_style='mmpose',
        #     show=False,
        #     wait_time=0,
        #     kpt_thr=0.3
        # )
        # end_time = time.time()
        # print(f"Function  executed in {end_time - start_time:.4f} seconds")
        # vis_img = self.visualizer.get_image()
        #
        # if vis_img is not None and not np.array_equal(vis_img, image):
        #
        #     return vis_img, pose_results
        # else:
        #     print("Visualization failed with official visualizer, falling back to manual drawing...")

        # Fallback: manual drawing

        vis_img = self._draw_poses_manually(image, pose_results)
        return vis_img, pose_results
        # Create a singleton instance


pose_service = PoseService()
