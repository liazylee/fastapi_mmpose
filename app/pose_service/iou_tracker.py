# app/pose_service/iou_tracker.py

import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class IoUTracker:
    """基于IoU的简单tracking算法"""
    
    def __init__(self, max_track_age: int = 60, iou_threshold: float = 0.5):
        """初始化IoU Tracker
        
        Args:
            max_track_age: tracks的最大保留帧数
            iou_threshold: IoU阈值用于tracking
        """
        # Tracking相关参数
        self.max_track_age = max_track_age
        self.iou_threshold = iou_threshold
        
        # Tracking状态
        self.track_history = []  # 存储历史tracks
        self.next_track_id = 1
        self.enable_tracking = True

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

    def track_persons(self, current_detections: List[Dict]) -> List[Dict]:
        """基于IoU的简单tracking算法
        
        Args:
            current_detections: 当前帧的检测结果列表，每个元素包含 'bbox', 'keypoints', 'scores'
            
        Returns:
            tracked_objects: 包含track_id的跟踪对象列表
        """
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