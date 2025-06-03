# app/pose_service/multi_person_onnx.py

import logging
from typing import List, Tuple, Dict

import cv2
import numpy as np
from mmdet.apis import inference_detector

from app.pose_service.onnx_inference import ONNXPoseEstimator
from app.pose_service.draw_pose_numba import draw_multi_person_pose_numba

logger = logging.getLogger(__name__)


class MultiPersonONNXPoseEstimator:
    """多人姿态估计器，结合MMDetection检测器和ONNX姿态模型，支持tracking"""

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

        # Tracking相关
        self.enable_tracking = True
        self.track_history = []  # 存储历史tracks
        self.next_track_id = 1
        self.max_track_age = 10  # tracks的最大保留帧数
        self.iou_threshold = 0.5  # IoU阈值用于tracking

    def set_detector(self, detector):
        """设置人体检测器"""
        self.detector = detector

    def reset_tracking(self):
        """重置tracking状态并清理内存"""
        logger.info(f"Resetting tracking - clearing {len(self.track_history)} track objects")
        
        # 清理tracking历史数据
        for track in self.track_history:
            # 清理numpy数组数据
            if 'keypoints' in track:
                track['keypoints'] = None
            if 'scores' in track:
                track['scores'] = None
            if 'bbox' in track:
                track['bbox'] = None
            track.clear()
        
        self.track_history.clear()
        self.next_track_id = 1
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        logger.info("Tracking reset completed with memory cleanup")

    def set_tracking_enabled(self, enabled: bool = True):
        """启用/禁用tracking"""
        self.enable_tracking = enabled
        if not enabled:
            self.reset_tracking()

    def _compute_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _track_persons(self, current_detections) -> List[Dict]:
        """基于IoU的简单tracking算法"""
        if not self.enable_tracking:
            # 如果不启用tracking，每个检测都分配新ID
            tracked_objects = []
            for i, detection in enumerate(current_detections):
                tracked_objects.append({
                    'track_id': i + 1,
                    'bbox': detection['bbox'],
                    'keypoints': detection['keypoints'],
                    'scores': detection['scores'],
                    'age': 0
                })
            return tracked_objects

        # 如果没有历史tracks，初始化
        if not self.track_history:
            tracked_objects = []
            for detection in current_detections:
                tracked_objects.append({
                    'track_id': self.next_track_id,
                    'bbox': detection['bbox'],
                    'keypoints': detection['keypoints'],
                    'scores': detection['scores'],
                    'age': 0
                })
                self.next_track_id += 1
            self.track_history = tracked_objects
            return tracked_objects

        # 计算IoU矩阵
        n_detections = len(current_detections)
        n_tracks = len(self.track_history)

        if n_detections == 0:
            # 没有新检测，增加所有tracks的age
            for track in self.track_history:
                track['age'] += 1
            # 移除过老的tracks
            self.track_history = [track for track in self.track_history if track['age'] < self.max_track_age]
            return self.track_history

        iou_matrix = np.zeros((n_detections, n_tracks))
        for i, detection in enumerate(current_detections):
            for j, track in enumerate(self.track_history):
                iou_matrix[i, j] = self._compute_iou(detection['bbox'], track['bbox'])

        # 贪心匹配：按IoU从大到小匹配
        matched_detections = set()
        matched_tracks = set()
        updated_tracks = []

        # 找到所有IoU > threshold的匹配
        matches = []
        for i in range(n_detections):
            for j in range(n_tracks):
                if iou_matrix[i, j] > self.iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))

        matches.sort(key=lambda x: x[2], reverse=True)

        # 进行匹配
        for det_idx, track_idx, iou_score in matches:
            if det_idx not in matched_detections and track_idx not in matched_tracks:
                # 更新现有track
                track = self.track_history[track_idx]
                track['bbox'] = current_detections[det_idx]['bbox']
                track['keypoints'] = current_detections[det_idx]['keypoints']
                track['scores'] = current_detections[det_idx]['scores']
                track['age'] = 0  # 重置age
                updated_tracks.append(track)
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)

        # 为未匹配的检测创建新tracks
        for i, detection in enumerate(current_detections):
            if i not in matched_detections:
                new_track = {
                    'track_id': self.next_track_id,
                    'bbox': detection['bbox'],
                    'keypoints': detection['keypoints'],
                    'scores': detection['scores'],
                    'age': 0
                }
                updated_tracks.append(new_track)
                self.next_track_id += 1

        # 保留未匹配但还年轻的tracks
        for j, track in enumerate(self.track_history):
            if j not in matched_tracks:
                track['age'] += 1
                if track['age'] < 5:  # 保留5帧
                    updated_tracks.append(track)

        self.track_history = updated_tracks
        return updated_tracks

    def inference(self, img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """对图像进行多人姿态估计和tracking

        Args:
            img: 输入图像，BGR格式

        Returns:
            tuple: (可视化图像, 关键点列表, 分数列表)
        """
        # 1. 人体检测
        bboxes = self._detect_humans(img)

        if len(bboxes) == 0:
            # 即使没有检测，也要更新tracking状态
            if self.enable_tracking:
                self._track_persons([])
            return img, [], []

        # 2. 批量姿态估计
        current_detections = []

        for bbox in bboxes:
            # 裁剪人体区域
            person_img = self._crop_person(img, bbox)

            # 姿态估计
            _, keypoints, scores = self.pose_estimator.inference(person_img)

            # 将关键点坐标转换回原图坐标系
            keypoints = self._transform_keypoints_to_original(keypoints, bbox)

            current_detections.append({
                'bbox': bbox[:4],  # [x1, y1, x2, y2]
                'keypoints': keypoints,
                'scores': scores
            })

        # 3. Tracking
        tracked_objects = self._track_persons(current_detections)

        # 4. 可视化
        vis_img = self._visualize_multi_person_with_tracking(img, tracked_objects)

        # 返回格式保持兼容
        keypoints_list = [obj['keypoints'] for obj in tracked_objects]
        scores_list = [obj['scores'] for obj in tracked_objects]

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

    def _visualize_multi_person_with_tracking(self, img: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        """可视化多人姿态和tracking ID - 使用Numba加速的绘制功能

        Args:
            img: 原始图像
            tracked_objects: 包含tracking信息的对象列表

        Returns:
            可视化后的图像
        """
        # 使用Numba加速的绘制函数
        return draw_multi_person_pose_numba(img, tracked_objects, self.pose_score_thr)
