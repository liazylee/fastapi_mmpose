import logging
import time

from aiortc import VideoStreamTrack

from app.pose_service.core import pose_service

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    """
    A VideoStreamTrack that transforms frames from an incoming track.
    """

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track  # incoming video track
        self.transform_enabled = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        logger.info("VideoTransformTrack initialized")

    async def recv(self):
        # Receive frame from upstream track
        frame = await self.track.recv()
        # Convert to numpy array (BGR format)
        # check the FPS of the incoming track
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        # Calculate FPS every second
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            logger.info(f"Current FPS: {self.fps:.2f}")
            # Reset counters
            self.frame_count = 0
            self.start_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        try:
            # Run MMPose processing: returns visualized image and pose results
            vis_img, pose_results = pose_service.process_image(img)
            if vis_img is not None:
                img = vis_img  # use visualized output
        except Exception as e:
            logger.error(f"MMPose processing error: {e}")
            # Fallback: keep original img

        # Build new frame from processed image
        new_frame = frame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def stop(self):
        """Stop the transform track"""
        self.transform_enabled = False
        if hasattr(self.track, 'stop'):
            self.track.stop()


# Global set to track peer connections
pcs = set()


def cleanup_pcs():
    """Helper function to clean up closed connections"""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
