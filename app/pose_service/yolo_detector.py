# app/pose_service/yolo_detector.py

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO检测器wrapper，支持检测和tracking功能"""

    def __init__(self, model_path: str = None, device: str = 'cuda', conf_threshold: float = 0.5):
        """初始化YOLO检测器
        
        Args:
            model_path: YOLO模型文件路径(.pt文件)
            device: 推理设备
            conf_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None

        # 如果提供了模型路径，立即加载
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载YOLO模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            # 导入ultralytics YOLO
            from ultralytics import YOLO

            self.model_path = model_path
            self.model = YOLO(model_path)
            self.model.to(self.device)

            logger.info(f"YOLO model loaded from {model_path} on {self.device}")

        except ImportError:
            logger.error("ultralytics package not found. Please install it: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect_and_track(self, img: np.ndarray, persist: bool = True) -> Tuple[List[np.ndarray], List[int]]:
        """使用YOLO进行检测和tracking
        
        Args:
            img: 输入图像，BGR格式
            persist: 是否持续tracking
            
        Returns:
            tuple: (检测框列表, track_id列表)
        """
        if self.model is None:
            logger.warning("YOLO model not loaded. Please call load_model() first.")
            return [], []

        try:
            # 使用YOLO的track方法进行检测和跟踪
            results = self.model.track(
                img,
                persist=persist,
                conf=self.conf_threshold,
                verbose=False
            )

            if not results or len(results) == 0:
                return [], []

            result = results[0]  # 取第一个结果

            # 提取检测框和track ID
            bboxes = []
            track_ids = []

            if result.boxes is not None and len(result.boxes) > 0:
                # 获取检测框坐标 [x1, y1, x2, y2]
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                # 获取置信度
                confidences = result.boxes.conf.cpu().numpy()
                # 获取类别ID（如果需要筛选特定类别）
                classes = result.boxes.cls.cpu().numpy()

                # 获取track ID（如果有的话）
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    # 如果没有track ID，使用索引作为临时ID
                    ids = list(range(len(boxes_xyxy)))

                # 组合检测结果
                for i, (box, conf, cls, track_id) in enumerate(zip(boxes_xyxy, confidences, classes, ids)):
                    # 添加置信度到检测框 [x1, y1, x2, y2, score]
                    bbox_with_score = np.append(box, conf)
                    bboxes.append(bbox_with_score)
                    track_ids.append(track_id)

            return bboxes, track_ids

        except Exception as e:
            logger.error(f"YOLO detection/tracking failed: {e}")
            return [], []

    def detect_only(self, img: np.ndarray) -> List[np.ndarray]:
        """仅进行检测，不包含tracking
        
        Args:
            img: 输入图像，BGR格式
            
        Returns:
            检测框列表 [[x1, y1, x2, y2, score], ...]
        """
        if self.model is None:
            logger.warning("YOLO model not loaded. Please call load_model() first.")
            return []

        try:
            # 使用YOLO的predict方法进行检测
            results = self.model.predict(
                img,
                conf=self.conf_threshold,
                verbose=False
            )

            if not results or len(results) == 0:
                return []

            result = results[0]  # 取第一个结果

            # 提取检测框
            bboxes = []

            if result.boxes is not None and len(result.boxes) > 0:
                # 获取检测框坐标 [x1, y1, x2, y2]
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                # 获取置信度
                confidences = result.boxes.conf.cpu().numpy()

                # 组合检测结果
                for box, conf in zip(boxes_xyxy, confidences):
                    # 添加置信度到检测框 [x1, y1, x2, y2, score]
                    bbox_with_score = np.append(box, conf)
                    bboxes.append(bbox_with_score)

            return bboxes

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def reset_tracking(self):
        """重置YOLO的tracking状态"""
        if self.model is not None:
            # YOLO的tracking状态重置
            if hasattr(self.model, 'trackers') and self.model.trackers:
                for tracker in self.model.trackers:
                    if hasattr(tracker, 'reset'):
                        tracker.reset()
            logger.info("YOLO tracking reset")

    def set_conf_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.conf_threshold = threshold

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {"status": "Model not loaded"}

        return {
            "model_path": self.model_path,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "model_type": type(self.model).__name__
        }


class YOLODetectorAdapter:
    """YOLO检测器适配器，提供与MMDetection兼容的接口"""

    def __init__(self, yolo_detector: YOLODetector):
        """初始化适配器
        
        Args:
            yolo_detector: YOLO检测器实例
        """
        self.yolo_detector = yolo_detector
        self.use_tracking = True  # 默认使用tracking

    def __call__(self, img: np.ndarray):
        """提供与MMDetection inference_detector兼容的调用接口"""
        if self.use_tracking:
            bboxes, track_ids = self.yolo_detector.detect_and_track(img)
            # 创建兼容的结果格式
            return YOLODetectionResult(bboxes, track_ids)
        else:
            bboxes = self.yolo_detector.detect_only(img)
            return YOLODetectionResult(bboxes, None)

    def set_tracking_enabled(self, enabled: bool):
        """启用/禁用tracking功能"""
        self.use_tracking = enabled


class YOLODetectionResult:
    """YOLO检测结果，提供与MMDetection兼容的接口"""

    def __init__(self, bboxes: List[np.ndarray], track_ids: Optional[List[int]] = None):
        """初始化检测结果
        
        Args:
            bboxes: 检测框列表
            track_ids: track ID列表（可选）
        """
        self.bboxes = np.array(bboxes) if bboxes else np.empty((0, 5))
        self.track_ids = track_ids

        # 创建pred_instances属性以兼容MMDetection格式
        self.pred_instances = YOLOPredInstances(self.bboxes, track_ids)


class YOLOPredInstances:
    """YOLO预测实例，模拟MMDetection的pred_instances格式"""

    def __init__(self, bboxes: np.ndarray, track_ids: Optional[List[int]] = None):
        """初始化预测实例
        
        Args:
            bboxes: 检测框数组 [N, 5] (x1, y1, x2, y2, score)
            track_ids: track ID列表
        """
        if len(bboxes) > 0:
            self.bboxes = bboxes[:, :4]  # [N, 4] (x1, y1, x2, y2)
            self.scores = bboxes[:, 4]  # [N] scores
            self.labels = np.zeros(len(bboxes), dtype=np.int64)  # 假设都是人(类别0)
        else:
            self.bboxes = np.empty((0, 4))
            self.scores = np.empty(0)
            self.labels = np.empty(0, dtype=np.int64)

        self.track_ids = track_ids

    def cpu(self):
        """返回CPU版本（用于兼容MMDetection接口）"""
        return MockTensor(self)

    def numpy(self):
        """返回numpy版本（用于兼容MMDetection接口）"""
        return self


class MockTensor:
    """模拟tensor的cpu()方法返回对象"""

    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data
