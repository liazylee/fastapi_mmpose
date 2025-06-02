import asyncio
import logging
from typing import Set

from aiortc import VideoStreamTrack
from av import VideoFrame

from app.config import batch_settings
from app.video_service.batch_processor import batch_processor
from app.video_service.permance_monitor import monitor

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    """应用姿态估计处理视频帧的视频轨道"""

    def __init__(self, track, config=None):
        super().__init__()
        self.track = track
        self.config = config or batch_settings
        self.active = True
        self.frame_counter = 0

        # 启动批处理器
        asyncio.create_task(batch_processor.start())

    async def recv(self):
        """处理每个带姿态估计的视频帧"""
        if not self.active:
            logger.warning("轨道停止后收到帧")
            return await self.track.recv()

        monitor.start_frame()
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        monitor.frame_received()

        # 生成唯一的帧ID
        frame_id = self.frame_counter
        self.frame_counter += 1

        # 通过批处理器处理
        vis_img = await batch_processor.process_frame(frame_id, img)

        if vis_img is not None:
            img = vis_img

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        monitor.frame_processed()
        return new_frame

    def stop(self):
        """停止处理并清理资源"""
        self.active = False
        logger.info("视频轨道已停止")

        # 停止批处理器
        asyncio.create_task(batch_processor.stop())

        # 清理连接管理
        cleanup_pcs()


# 连接管理
pcs: Set = set()


def cleanup_pcs():
    """从集合中移除已关闭的连接"""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
