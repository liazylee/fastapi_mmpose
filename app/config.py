import os
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "FastAPI MMPose"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-here"
    DATABASE_URL: str = "sqlite:///./app.db"

    # Get the project root directory
    PROJECT_ROOT: ClassVar[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Model configuration paths
    DETECTOR_CONFIG: str = os.path.join(PROJECT_ROOT, "app/pose_service/configs/rtmdet_m_640-8xb32_coco-person.py")
    DETECTOR_CHECKPOINT: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

    POSE_CONFIG: str = os.path.join(PROJECT_ROOT, "app/pose_service/configs/rtmpose-m_8xb256-420e_body8-256x192.py")
    POSE_CHECKPOINT: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'

    # Device configuration
    DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    RADIUS: int = 3
    ALPHA: float = 0.8
    LINE_WIDTH: int = 1
    det_cat_id: int = 0  # Category ID for person detection
    bbox_thr: float = 0.5  # Threshold for bounding box confidence
    nms_thr: float = 0.4  # IoU threshold for NMS
    FPS: int = 30  # Default FPS for video processing
    WORKERS: int = os.cpu_count() or 4  # Number of CPU workers

    class Config:
        case_sensitive = True


@dataclass
class BatchProcessingConfig:
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    PROJECT_ROOT: ClassVar[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Batch settings - optimized for real-time video streaming
    batch_size: int = 16  # Reduced from 32 to 8 for faster batch formation in real-time streams
    max_queue_size: int = 50
    batch_timeout_ms: int = 500  # Reduced from 100ms to 50ms for lower latency

    # Threading settings
    num_workers: int = os.cpu_count() or 4
    gpu_streams: int = 40

    # Model settings
    enable_tensorrt: bool = False
    input_resolution: tuple = (1920, 1080)
    detection_batch_size: int = 4
    pose_batch_size: int = 10
    det_score_thr: float = 0.4
    pose_score_thr: float = 0.5
    max_track_age: int = 60
    iou_threshold: float = 0.5
    onnx_path = os.path.join(PROJECT_ROOT, "pose_service/configs/rtmpose_onnx/end2end.onnx")

    # YOLO Detection and Tracking settings
    detector_type: str = 'yolo'
    yolo_model_path: str = os.path.join(PROJECT_ROOT, "app/pose_service/configs/yolo11n.pt")
    yolo_conf_threshold: float = 0.5
    yolo_device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    yolo_iou_threshold: float = 0.7
    yolo_max_det: int = 300
    yolo_classes: list = field(default_factory=lambda: [0])  # Only detect persons

    # YOLO Tracking settings
    use_yolo_detector: bool = True
    enable_tracking: bool = True
    use_yolo_tracking: bool = False  # 是否使用YOLO内置tracking，现在设为False使用我们的简化实现
    use_bytetrack: bool = True  # 是否使用ByteTrack（通过YOLO）
    tracker_type: str = 'bytetrack'  # YOLO支持的tracker类型: bytetrack, botsort
    track_persist: bool = True  # 持续跟踪
    track_verbose: bool = False  # 跟踪输出详细信息


batch_settings: BatchProcessingConfig = BatchProcessingConfig()
settings: Settings = Settings()
