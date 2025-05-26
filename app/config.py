import os
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
    bbox_thr: float = 0.3  # Threshold for bounding box confidence
    nms_thr: float = 0.3  # IoU threshold for NMS

    class Config:
        case_sensitive = True


settings = Settings()
