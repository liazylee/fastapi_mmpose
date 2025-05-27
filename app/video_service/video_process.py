import logging
import time
from collections import deque

from aiortc import VideoStreamTrack
from av import VideoFrame

from app.pose_service.core import pose_service

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    """Video transformation track for pose detection processing"""

    def __init__(self, track, enable_skip=True):
        super().__init__()
        self.track = track
        self.transform_enabled = True
        self.enable_skip = enable_skip

        # Performance statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()

        # Processing time statistics
        self.process_times = deque(maxlen=30)
        self.recv_times = deque(maxlen=30)

        # Frame skipping control
        self.processing = False
        self.last_processed_frame = None
        self.skip_count = 0

        logger.info(f"VideoTransformTrack initialized, enable_skip={enable_skip}")

    async def recv(self):
        # Receive upstream frame
        recv_start = time.time()
        frame = await self.track.recv()
        recv_time = time.time() - recv_start
        self.recv_times.append(recv_time)

        # Update statistics
        self.frame_count += 1

        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Use previous frame if currently processing and skip is enabled
        if self.enable_skip and self.processing and self.last_processed_frame is not None:
            self.skip_count += 1
            img = self.last_processed_frame
        elif self.transform_enabled:
            self.processing = True
            process_start = time.time()

            vis_img, pose_results = pose_service.process_image(img)

            process_time = time.time() - process_start
            self.process_times.append(process_time)

            if vis_img is not None:
                img = vis_img
                self.last_processed_frame = img

            self.processing = False

        # Log performance metrics every second
        current_time = time.time()
        elapsed_time = current_time - self.last_log_time
        if elapsed_time > 1.0:
            fps = self.frame_count / elapsed_time
            avg_recv_time = sum(self.recv_times) / max(len(self.recv_times), 1)
            avg_process_time = sum(self.process_times) / max(len(self.process_times), 1)
            skip_rate = (self.skip_count / max(self.frame_count, 1) * 100)

            logger.info(
                f"Transform FPS: {fps:.1f}, Recv: {avg_recv_time * 1000:.1f}ms, "
                f"Process: {avg_process_time * 1000:.1f}ms, Skip rate: {skip_rate:.1f}%"
            )

            # Reset counters
            self.frame_count = 0
            self.skip_count = 0
            self.last_log_time = current_time

        # Create new frame
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame

    def stop(self):
        """Stop processing track"""
        self.transform_enabled = False
        if hasattr(self.track, 'stop'):
            self.track.stop()


# Connection management
pcs = set()


def cleanup_pcs():
    """Remove closed connections from the set"""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
