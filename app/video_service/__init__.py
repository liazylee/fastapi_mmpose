from .async_pose_service import AsyncPoseService

# 添加新的ONNX导入
from .batch_GPU_process_onnx import BatchPoseProcessorONNX
from .video_process import VideoTransformTrack

# 导出所有可用的类
__all__ = [
    # 原有的

    'AsyncPoseService',
    'VideoTransformTrack',
    # 新的ONNX版本
    'BatchPoseProcessorONNX',

]
