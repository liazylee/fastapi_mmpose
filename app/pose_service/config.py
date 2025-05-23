import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Detection model configs
DET_CONFIG = os.path.join(PROJECT_ROOT, "configs/minimal_det_config.py")
DET_CHECKPOINT = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

# Pose model configs
POSE_CONFIG = os.path.join(PROJECT_ROOT, "configs/minimal_pose_config.py")
POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"

# Device settings
DEVICE = "cpu"  # Use CPU for stability

# Detection settings
DET_CAT_ID = 0  # COCO person category id
BBOX_THR = 0.3
NMS_THR = 0.3

# Pose estimation settings
KPT_THR = 0.3
MIN_BBOX_SIZE = 32
MAX_NUM_PERSON = 30

# Visualization settings
DRAW_HEATMAP = False
SHOW_KPT_IDX = False
SKELETON_STYLE = "mmpose"  # choices: ['mmpose', 'openpose']
RADIUS = 3
THICKNESS = 1
ALPHA = 0.8
DRAW_BBOX = True
