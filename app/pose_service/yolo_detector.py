# app/pose_service/yolo_detector.py

import logging
from typing import List, Dict, Optional, Union

import numpy as np
from ultralytics import YOLO

from app.config import batch_settings

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO检测器，支持批量检测功能"""

    def __init__(self):
        self.config = batch_settings
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        model = YOLO(self.config.yolo_model_path)
        model.to(self.config.yolo_device)
        logger.info(f"YOLO model loaded: {self.config.yolo_model_path}")
        return model

    def detect_batch(self, imgs: List[np.ndarray]) -> List[List[Dict]]:
        """批量检测，返回标准化的检测结果"""
        if not imgs:
            return []

        results = self.model.predict(
            imgs,
            conf=self.config.yolo_conf_threshold,
            iou=self.config.yolo_iou_threshold,
            max_det=self.config.yolo_max_det,
            classes=self.config.yolo_classes,
            verbose=self.config.track_verbose,
            device=self.config.yolo_device
        )

        return [self._format_detections(result) for result in results]

    def detect_single(self, img: np.ndarray) -> List[Dict]:
        """单张图像检测"""
        batch_results = self.detect_batch([img])
        return batch_results[0] if batch_results else []

    def _format_detections(self, result) -> List[Dict]:
        """格式化检测结果"""
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                detections.append({
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'score': float(conf),
                    'class_id': int(cls)
                })

        return detections

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_path": self.config.yolo_model_path,
            "device": self.config.yolo_device,
            "conf_threshold": self.config.yolo_conf_threshold,
            "iou_threshold": self.config.yolo_iou_threshold
        }


class YOLOTracker:
    """YOLO跟踪器，使用YOLO自带的tracking功能"""

    def __init__(self):
        self.config = batch_settings
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        model = YOLO(self.config.yolo_model_path)
        model.to(self.config.yolo_device)
        logger.info(f"YOLO tracker loaded: {self.config.yolo_model_path}")
        return model

    def track_single(self, img: np.ndarray) -> List[Dict]:
        """单帧跟踪"""
        results = self.model.track(
            img,
            conf=self.config.yolo_conf_threshold,
            iou=self.config.yolo_iou_threshold,
            max_det=self.config.yolo_max_det,
            classes=self.config.yolo_classes,
            tracker=f"{self.config.tracker_type}.yaml",
            persist=self.config.track_persist,
            verbose=self.config.track_verbose,
            device=self.config.yolo_device
        )

        return self._format_tracks(results[0]) if results else []

    def _format_tracks(self, result) -> List[Dict]:
        """格式化跟踪结果"""
        tracks = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id

            # 处理track_ids可能为None的情况
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)
            else:
                track_ids = list(range(len(boxes)))

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                tracks.append({
                    'track_id': int(track_ids[i]),
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'score': float(conf),
                    'class_id': int(cls)
                })

        return tracks

    def reset_tracker(self):
        """重置跟踪器"""
        # YOLO的跟踪器会自动处理状态，无需手动重置
        logger.info("YOLO tracker reset")


class DetectorWithTracking:
    """结合检测和跟踪的完整解决方案"""

    def __init__(self):
        self.config = batch_settings
        self.detector = YOLODetector()
        self.tracker = YOLOTracker() if self.config.enable_tracking else None

    def detect_and_track_batch(self, imgs: List[np.ndarray]) -> List[List[Dict]]:
        """批量检测 + 逐帧跟踪"""
        if not imgs:
            return []

        if self.tracker:
            # 使用跟踪模式（逐帧处理以保持时序状态）
            return [self.tracker.track_single(img) for img in imgs]
        else:
            # 仅检测模式（可以批量处理）
            return self.detector.detect_batch(imgs)

    def detect_and_track_single(self, img: np.ndarray) -> List[Dict]:
        """单帧检测和跟踪"""
        if self.tracker:
            return self.tracker.track_single(img)
        else:
            return self.detector.detect_single(img)

    def process_video_stream(self, imgs: List[np.ndarray]) -> List[List[Dict]]:
        """处理视频流"""
        return self.detect_and_track_batch(imgs)

    def reset_tracker(self):
        """重置跟踪器"""
        if self.tracker:
            self.tracker.reset_tracker()

    def get_info(self) -> Dict:
        """获取系统信息"""
        info = {
            "detector": self.detector.get_model_info(),
            "tracking_enabled": self.tracker is not None,
            "tracker_type": self.config.tracker_type if self.tracker else None
        }
        return info


# 向后兼容适配器
class YOLODetectorAdapter:
    """适配器类，提供与现有代码兼容的接口"""

    def __init__(self):
        self.detector_tracker = DetectorWithTracking()

    def __call__(self, img: np.ndarray):
        """兼容现有调用接口"""
        tracks = self.detector_tracker.detect_and_track_single(img)
        return self._create_detection_result(tracks)

    def _create_detection_result(self, tracks: List[Dict]):
        """创建兼容的检测结果"""
        if not tracks:
            return DetectionResult([], [])

        bboxes = []
        track_ids = []

        for track in tracks:
            bbox = track['bbox'] + [track['score']]  # [x1, y1, x2, y2, score]
            bboxes.append(np.array(bbox))
            track_ids.append(track.get('track_id', 0))

        return DetectionResult(bboxes, track_ids)


class DetectionResult:
    """检测结果类，兼容现有接口"""

    def __init__(self, bboxes: List[np.ndarray], track_ids: Optional[List[int]] = None):
        self.bboxes = np.array(bboxes) if bboxes else np.empty((0, 5))
        self.track_ids = track_ids or []
        self.pred_instances = PredInstances(self.bboxes, track_ids)


class PredInstances:
    """预测实例类，兼容现有接口"""

    def __init__(self, bboxes: np.ndarray, track_ids: Optional[List[int]] = None):
        if len(bboxes) > 0:
            self.bboxes = bboxes[:, :4]  # [N, 4] (x1, y1, x2, y2)
            self.scores = bboxes[:, 4]  # [N] scores
            self.labels = np.zeros(len(bboxes), dtype=np.int64)
        else:
            self.bboxes = np.empty((0, 4))
            self.scores = np.empty(0)
            self.labels = np.empty(0, dtype=np.int64)

        self.track_ids = track_ids or []

    def cpu(self):
        return MockTensor(self)

    def numpy(self):
        return self


class MockTensor:
    """模拟tensor接口"""

    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data
