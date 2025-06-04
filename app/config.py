import os
from dataclasses import dataclass
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
    bbox_thr: float = 0.4  # Threshold for bounding box confidence
    nms_thr: float = 0.4  # IoU threshold for NMS
    FPS: int = 30  # Default FPS for video processing
    WORKERS: int = os.cpu_count() or 4  # Number of CPU workers

    class Config:
        case_sensitive = True


@dataclass
class BatchProcessingConfig:
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    PROJECT_ROOT: ClassVar[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Batch settings
    batch_size: int = 16
    max_queue_size: int = 50
    batch_timeout_ms: int = 2  # Max wait time to form batch

    # Threading settings
    num_workers: int = os.cpu_count() or 4  # Separate threads for different stages
    gpu_streams: int = 40  # CUDA streams for parallel processing

    # # Performance settings
    # prefetch_factor: int = 20
    # pin_memory: bool = True
    # non_blocking: bool = True

    # Model settings
    enable_tensorrt: bool = False  # Toggle for future TensorRT
    input_resolution: tuple = (1920, 1080)  # Input resolution for video processing
    detection_batch_size: int = 16  # Can differ from pose batch size
    pose_batch_size: int = 10
    det_score_thr: float = 0.3
    pose_score_thr: float = 0.3
    max_track_age: int = 60
    iou_threshold: float = 0.5
    onnx_path = os.path.join(PROJECT_ROOT,
                             "pose_service/configs/rtmpose_onnx/end2end.onnx")


batch_settings: BatchProcessingConfig = BatchProcessingConfig()

settings: Settings = Settings()
