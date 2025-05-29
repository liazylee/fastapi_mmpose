import logging

from aiortc import VideoStreamTrack
from av import VideoFrame

from app.config import batch_settings
from app.video_service.async_pose_service import AsyncPoseService
from app.video_service.permance_monitor import monitor

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, config=None):
        super().__init__()
        self.track = track
        self.config = config or batch_settings
        # Use async pose service
        self.pose_service = AsyncPoseService(self.config)

    async def recv(self):
        monitor.start_frame()
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        monitor.frame_received()
        # Process through async pipeline
        vis_img, pose_results = await self.pose_service.process_frame_async(img)

        if vis_img is not None:
            img = vis_img

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        monitor.frame_processed()
        return new_frame

    def stop(self):
        """停止处理服务"""
        if hasattr(self.pose_service, 'stop_pipeline'):
            # AsyncPoseService
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.pose_service.stop_pipeline())
            except Exception as e:
                logger.error(f"Error stopping AsyncPoseService: {e}")
        elif hasattr(self.pose_service, 'stop'):
            # ThreadedPoseService
            self.pose_service.stop()

        logger.info(f"Stopped pose service: {self.pose_service}")
        logger.warning("recv exited due to stop()")

    def __del__(self):
        """析构时清理资源"""
        try:
            self.stop()
        except:
            pass


# Connection management
pcs = set()


def cleanup_pcs():
    """Remove closed connections from the set"""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
