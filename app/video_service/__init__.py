# 添加新的ONNX导入
from .batch_GPU_process_onnx import BatchPoseProcessorONNX
from .video_process import VideoTransformTrack

__all__ = [

    'VideoTransformTrack',
    # 新的ONNX版本
    'BatchPoseProcessorONNX',

]
