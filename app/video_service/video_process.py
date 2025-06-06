import asyncio
import logging
from typing import Optional

from aiortc import VideoStreamTrack
from av import VideoFrame

from app.config import batch_settings
from app.video_service.streaming_processor import StreamingPoseProcessor

logger = logging.getLogger(__name__)


class AsyncVideoTransformTrack(VideoStreamTrack):
    """
    Asynchronous video transform track with non-blocking processing
    Implements the new streaming architecture:
    - recv() only collects frames + returns latest processed frame
    - Background batch processing runs independently
    - Controlled output timing for stable FPS
    """

    def __init__(self, track, config=None, target_fps: int = 30):
        super().__init__()
        self.track = track
        self.config = config or batch_settings
        self.target_fps = target_fps

        # Initialize the streaming processor
        self.processor = StreamingPoseProcessor(self.config, target_fps)

        # Frame management
        self.last_output_frame: Optional[VideoFrame] = None
        self.frame_count = 0

        # Performance tracking
        self.processing_enabled = True
        self._is_started = False

        # Simple track state
        self._track_ended = False

        logger.info(f"AsyncVideoTransformTrack initialized - target FPS: {target_fps}")

    async def _ensure_processor_started(self):
        """Ensure the processor is started"""
        if not self._is_started:
            await self.processor.start()
            self._is_started = True

    async def recv(self):
        """Non-blocking recv() - ends when track ends"""
        """优化的 recv 方法 - 减少跳帧和卡顿"""
        if self._track_ended:
            raise Exception("Video track has ended")

        await self._ensure_processor_started()

        # Get source frame
        source_frame = await self.track.recv()
        input_image = source_frame.to_ndarray(format="bgr24")

        # Add to processing pipeline
        if self.processing_enabled:
            self.processor.add_frame(input_image)

        # 异步获取输出帧，带等待
        output_frame, should_output = await self.processor.get_output_frame_async(timeout_ms=50)

        if should_output and output_frame is not None:
            # New processed frame
            self.last_output_frame = VideoFrame.from_ndarray(output_frame, format="bgr24")
            self.last_output_frame.pts = source_frame.pts
            self.last_output_frame.time_base = source_frame.time_base
            self._save_timing_info(source_frame)
            self.frame_count += 1
            return self.last_output_frame

        elif self.last_output_frame is not None:
            # Repeat last frame with new timing (减少重复帧的使用)
            # 只有在等待超时时才使用重复帧
            repeated_frame = VideoFrame.from_ndarray(
                self.last_output_frame.to_ndarray(format="bgr24"), format="bgr24"
            )
            repeated_frame.pts = source_frame.pts
            repeated_frame.time_base = source_frame.time_base
            self._save_timing_info(source_frame)
            return repeated_frame

        else:
            # Return original frame
            self.last_output_frame = source_frame
            self._save_timing_info(source_frame)
            return self.last_output_frame

    def _save_timing_info(self, frame):
        """Save timing information"""
        self._last_pts = frame.pts
        self._last_time_base = frame.time_base

    def enable_processing(self):
        """Enable pose processing"""
        self.processing_enabled = True
        logger.info("Pose processing enabled")

    def disable_processing(self):
        """Disable pose processing - will output original frames"""
        self.processing_enabled = False
        logger.info("Pose processing disabled")

    def reset_tracking(self):
        """Reset tracking functionality"""
        if hasattr(self.processor, 'batch_processor') and \
                hasattr(self.processor.batch_processor, 'processor'):
            processor = self.processor.batch_processor.processor
            if hasattr(processor, 'reset_tracking'):
                processor.reset_tracking()
            if hasattr(processor, 'pose_estimator') and \
                    hasattr(processor.pose_estimator, 'reset_tracking'):
                processor.pose_estimator.reset_tracking()
        logger.info("Tracking reset in async video track")

    def set_tracking_enabled(self, enabled: bool):
        """Enable/disable tracking functionality"""
        if hasattr(self.processor, 'batch_processor') and \
                hasattr(self.processor.batch_processor, 'processor'):
            processor = self.processor.batch_processor.processor
            if hasattr(processor, 'set_tracking_enabled'):
                processor.set_tracking_enabled(enabled)
            if hasattr(processor, 'pose_estimator') and \
                    hasattr(processor.pose_estimator, 'set_tracking_enabled'):
                processor.pose_estimator.set_tracking_enabled(enabled)
        logger.info(f"Tracking {'enabled' if enabled else 'disabled'} in async video track")

    def is_tracking_enabled(self) -> bool:
        """Check if tracking is enabled"""
        return self.processing_enabled

    async def stop(self):
        """Stop the video track and clean up resources"""
        logger.info("Stopping async video transform track...")

        # Stop the streaming processor
        if self._is_started:
            await self.processor.stop()
            self._is_started = False

        # Clear references
        self.processor = None
        self.track = None
        self.last_output_frame = None

        logger.info("Async video transform track stopped")

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics"""
        if hasattr(self.processor, 'get_comprehensive_stats'):
            return self.processor.get_comprehensive_stats()
        return {}

    def __del__(self):
        """Cleanup on destruction"""
        if self._is_started and self.processor is not None:
            # Schedule cleanup task if event loop is available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task to stop processor
                    asyncio.create_task(self._cleanup_async())
                else:
                    # If no running loop, run cleanup synchronously
                    asyncio.run(self._cleanup_async())
            except RuntimeError:
                # No event loop available, do synchronous cleanup
                self._cleanup_sync()

    async def _cleanup_async(self):
        """Async cleanup helper"""
        if self._is_started and self.processor is not None:
            await self.processor.stop()
            self._is_started = False
            self.processor = None

    def _cleanup_sync(self):
        """Synchronous cleanup helper for when no event loop is available"""
        if self.processor is not None:
            # Mark as stopped
            self._is_started = False
            # Clear references without async cleanup
            self.processor = None
            self.track = None
            self.last_output_frame = None


# For backward compatibility, provide the original class name
class VideoTransformTrack(AsyncVideoTransformTrack):
    """Backward compatibility alias"""
    pass


# Global management functions remain the same
pcs = set()
_global_tracking_enabled = True
_active_video_tracks = []


def get_global_tracking_status():
    """Get global tracking status"""
    return {
        "enabled": _global_tracking_enabled,
        "active_tracks": len(_active_video_tracks)
    }


def set_global_tracking_enabled(enabled: bool):
    """Set global tracking status"""
    global _global_tracking_enabled
    _global_tracking_enabled = enabled

    # Update all active video tracks
    for video_track in _active_video_tracks:
        if hasattr(video_track, 'set_tracking_enabled'):
            video_track.set_tracking_enabled(enabled)

    logger.info(f"Global tracking {'enabled' if enabled else 'disabled'} for {len(_active_video_tracks)} tracks")


def reset_global_tracking():
    """Reset all active video tracks' tracking"""
    for video_track in _active_video_tracks:
        if hasattr(video_track, 'reset_tracking'):
            video_track.reset_tracking()

    logger.info(f"Global tracking reset for {len(_active_video_tracks)} tracks")


def register_video_track(video_track):
    """Register new video track"""
    if video_track not in _active_video_tracks:
        _active_video_tracks.append(video_track)
        # Apply global tracking settings
        if hasattr(video_track, 'set_tracking_enabled'):
            video_track.set_tracking_enabled(_global_tracking_enabled)


def unregister_video_track(video_track):
    """Unregister video track"""
    if video_track in _active_video_tracks:
        _active_video_tracks.remove(video_track)


def cleanup_pcs():
    """Remove closed connections from the set"""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
