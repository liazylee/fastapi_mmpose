# app/pose_service/multi_person_onnx.py

import logging
from typing import List, Tuple

import cv2
import numpy as np
from mmdet.apis import inference_detector

from app.pose_service.onnx_inference import ONNXPoseEstimator

logger = logging.getLogger(__name__)


class MultiPersonONNXPoseEstimator:
    """多人姿态估计器，结合MMDetection检测器和ONNX姿态模型"""

    def __init__(self, onnx_file: str, device: str = 'cuda'):
        """初始化多人姿态估计器

        Args:
            onnx_file: ONNX模型文件路径
            device: 推理设备
        """
        self.device = device
        self.pose_estimator = ONNXPoseEstimator(onnx_file, device)
        self.detector = None  # 将在外部设置

        # 检测和姿态估计的阈值
        self.det_score_thr = 0.5
        self.pose_score_thr = 0.3

    def set_detector(self, detector):
        """设置人体检测器"""
        self.detector = detector

    def inference(self, img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """对图像进行多人姿态估计

        Args:
            img: 输入图像，BGR格式

        Returns:
            tuple: (可视化图像, 关键点列表, 分数列表)
        """
        # 1. 人体检测
        bboxes = self._detect_humans(img)

        if len(bboxes) == 0:
            return img, [], []

        # 2. 批量姿态估计
        keypoints_list = []
        scores_list = []

        for bbox in bboxes:
            # 裁剪人体区域
            person_img = self._crop_person(img, bbox)

            # 姿态估计
            _, keypoints, scores = self.pose_estimator.inference(person_img)

            # 将关键点坐标转换回原图坐标系
            keypoints = self._transform_keypoints_to_original(keypoints, bbox)

            keypoints_list.append(keypoints)
            scores_list.append(scores)

        # 3. 可视化
        vis_img = self._visualize_multi_person(img, keypoints_list, scores_list)

        return vis_img, keypoints_list, scores_list

    def _detect_humans(self, img: np.ndarray) -> List[np.ndarray]:
        """检测图像中的人体

        Returns:
            人体边界框列表 [[x1, y1, x2, y2, score], ...]
        """
        if self.detector is None:
            logger.warning("检测器未设置，跳过人体检测")
            # 返回整张图作为默认bbox
            h, w = img.shape[:2]
            return [np.array([0, 0, w, h, 1.0])]

        # 使用MMDetection进行检测
        result = inference_detector(self.detector, img)

        # 提取人体类别的检测结果（COCO中人是第0类）
        if hasattr(result, 'pred_instances'):
            # MMDetection v3.x格式
            instances = result.pred_instances
            labels = instances.labels.cpu().numpy()
            bboxes = instances.bboxes.cpu().numpy()
            scores = instances.scores.cpu().numpy()

            # 筛选人体检测结果
            person_indices = np.where((labels == 0) & (scores > self.det_score_thr))[0]
            person_bboxes = []

            for idx in person_indices:
                bbox = bboxes[idx]
                score = scores[idx]
                person_bboxes.append(np.append(bbox, score))
        else:
            # 旧版本MMDetection格式
            bboxes = result[0]  # 第0类是人
            person_bboxes = [bbox for bbox in bboxes if bbox[4] > self.det_score_thr]

        return person_bboxes

    def _crop_person(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """根据边界框裁剪人体区域

        Args:
            img: 原始图像
            bbox: 边界框 [x1, y1, x2, y2, score]

        Returns:
            裁剪后的人体图像
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)

        # 添加边界检查
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 扩展边界框以包含更多上下文
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 扩展20%
        new_w = int(bbox_w * 1.2)
        new_h = int(bbox_h * 1.2)

        new_x1 = max(0, cx - new_w // 2)
        new_y1 = max(0, cy - new_h // 2)
        new_x2 = min(w, cx + new_w // 2)
        new_y2 = min(h, cy + new_h // 2)

        return img[new_y1:new_y2, new_x1:new_x2]

    def _transform_keypoints_to_original(self, keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """将关键点坐标从裁剪图像坐标系转换回原图坐标系

        Args:
            keypoints: 裁剪图像中的关键点坐标
            bbox: 原始边界框

        Returns:
            原图坐标系中的关键点
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)

        # 计算扩展后的边界框
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        new_w = int(bbox_w * 1.2)
        new_h = int(bbox_h * 1.2)

        new_x1 = cx - new_w // 2
        new_y1 = cy - new_h // 2

        # 转换坐标
        transformed_keypoints = keypoints.copy()
        if len(transformed_keypoints.shape) == 3:
            # 批量处理
            transformed_keypoints[:, :, 0] += new_x1
            transformed_keypoints[:, :, 1] += new_y1
        else:
            # 单个关键点集
            transformed_keypoints[:, 0] += new_x1
            transformed_keypoints[:, 1] += new_y1

        return transformed_keypoints

    def _visualize_multi_person(self, img: np.ndarray, keypoints_list: List[np.ndarray],
                                scores_list: List[np.ndarray]) -> np.ndarray:
        """可视化多人姿态

        Args:
            img: 原始图像
            keypoints_list: 所有人的关键点列表
            scores_list: 所有人的关键点分数列表

        Returns:
            可视化后的图像
        """
        vis_img = img.copy()

        # 为每个人使用不同的颜色
        person_colors = [
            (255, 0, 0),  # 红
            (0, 255, 0),  # 绿
            (0, 0, 255),  # 蓝
            (255, 255, 0),  # 黄
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 青
        ]

        for person_idx, (keypoints, scores) in enumerate(zip(keypoints_list, scores_list)):
            # 选择颜色
            color_base = person_colors[person_idx % len(person_colors)]

            # 确保keypoints是2D数组
            if len(keypoints.shape) == 3 and keypoints.shape[0] == 1:
                keypoints = keypoints[0]
            if len(scores.shape) == 2 and scores.shape[0] == 1:
                scores = scores[0]

            # 绘制骨架
            skeleton = [
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]
            ]

            # 绘制连接线
            for connection in skeleton:
                kpt_a, kpt_b = connection
                if kpt_a < len(keypoints) and kpt_b < len(keypoints):
                    if scores[kpt_a] > self.pose_score_thr and scores[kpt_b] > self.pose_score_thr:
                        pos_a = tuple(keypoints[kpt_a].astype(int))
                        pos_b = tuple(keypoints[kpt_b].astype(int))
                        cv2.line(vis_img, pos_a, pos_b, color_base, 2)

            # 绘制关键点
            for kpt_idx, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score > self.pose_score_thr:
                    pos = tuple(kpt.astype(int))
                    cv2.circle(vis_img, pos, 4, color_base, -1)
                    cv2.circle(vis_img, pos, 5, (255, 255, 255), 1)

        return vis_img
