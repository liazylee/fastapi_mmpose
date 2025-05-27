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

    def _draw_poses_manually(self, image, pose_results):
        """手动绘制关键点，作为备用方案"""
        vis_img = image.copy()

        # 获取骨架连接信息
        skeleton = self.pose_estimator.dataset_meta.get('skeleton_info', {}).get('skeleton', [])
        joint_weights = self.pose_estimator.dataset_meta.get('joint_weights', [])

        for pose_result in pose_results:
            if hasattr(pose_result, 'pred_instances'):
                keypoints = pose_result.pred_instances.keypoints[0]  # 取第一个人
                scores = pose_result.pred_instances.keypoint_scores[0]

                # 绘制关键点
                for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                    if score > 0.3:  # 置信度阈值
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
                        cv2.putText(vis_img, str(i), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # 绘制骨架连接
                for connection in skeleton:
                    pt1_idx, pt2_idx = connection
                    if (scores[pt1_idx] > 0.3 and scores[pt2_idx] > 0.3):
                        pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                        pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                        cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)

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

        data_samples = merge_data_samples(pose_results)
        # Method 1: Use add_datasample

        start_time = time.time()
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

        self.visualizer.add_datasample(
            'result',
            image,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,  # Do not draw heatmap
            draw_bbox=True,
            show_kpt_idx=True,
            skeleton_style='mmpose',
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )
        end_time = time.time()
        print(f"Function  executed in {end_time - start_time:.4f} seconds")
        vis_img = self.visualizer.get_image()

        if vis_img is not None and not np.array_equal(vis_img, image):

            return vis_img, pose_results
        else:
            print("Visualization failed with official visualizer, falling back to manual drawing...")

        # Fallback: manual drawing

        vis_img = self._draw_poses_manually(image, pose_results)
        return vis_img, pose_results
        # Create a singleton instance


pose_service = PoseService()
