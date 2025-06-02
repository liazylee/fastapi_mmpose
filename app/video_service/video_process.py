import asyncio
import logging
from typing import Set

from aiortc import VideoStreamTrack
from av import VideoFrame

from app.config import batch_settings
# from app.video_service.async_pose_service import AsyncPoseService
from app.video_service.permance_monitor import monitor
from app.video_service.rtlib import rtlib_service

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    """Video track that applies pose estimation to each frame."""
    
    def __init__(self, track, config=None):
        super().__init__()
        self.track = track
        self.config = config or batch_settings
        self.active = True
        # Use async pose service
        self.pose_service = rtlib_service

    async def recv(self):
        """Process each video frame with pose estimation."""
        if not self.active:
            logger.warning("Received frame after track was stopped")
            return await self.track.recv()
        
        monitor.start_frame()
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        monitor.frame_received()
        
        # Process through RTLib pipeline
        vis_img = await self.pose_service.predict_vis(img)
        if vis_img is not None:
            img = vis_img

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        monitor.frame_processed()
        return new_frame




# Connection management
pcs: Set = set()


def cleanup_pcs():
    """Remove closed connections from the set."""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
