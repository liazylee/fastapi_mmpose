import logging

from aiortc import VideoStreamTrack
from av import VideoFrame

from app.config import batch_settings
from app.video_service.async_pose_service import AsyncPoseService

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, config=None):
        super().__init__()
        self.track = track
        self.config = config or batch_settings

        # Use async pose service
        self.pose_service = AsyncPoseService(self.config)

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Process through async pipeline
        vis_img, pose_results = await self.pose_service.process_frame_async(img)

        if vis_img is not None:
            img = vis_img

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame


# Connection management
pcs = set()


def cleanup_pcs():
    """Remove closed connections from the set"""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
