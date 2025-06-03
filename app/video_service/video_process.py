import logging

from aiortc import VideoStreamTrack
from av import VideoFrame

from app.config import batch_settings
from app.video_service.async_pose_service import async_pose_service
from app.video_service.permance_monitor import monitor

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, config=None):
        super().__init__()
        self.track = track
        self.config = config or batch_settings

        # Use async pose service
        self.pose_service = async_pose_service
        
        # Tracking enabled by default
        self._tracking_enabled = True

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

    def reset_tracking(self):
        """重置tracking功能"""
        try:
            # 重置async pose service中的tracking
            if hasattr(self.pose_service, 'processor') and hasattr(self.pose_service.processor, 'reset_tracking'):
                self.pose_service.processor.reset_tracking()
            # 重置batch processor中的tracking  
            if hasattr(self.pose_service, 'processor') and hasattr(self.pose_service.processor, 'pose_estimator'):
                if hasattr(self.pose_service.processor.pose_estimator, 'reset_tracking'):
                    self.pose_service.processor.pose_estimator.reset_tracking()
            logger.info("Tracking reset in video transform track")
        except Exception as e:
            logger.warning(f"Failed to reset tracking: {e}")

    def set_tracking_enabled(self, enabled: bool):
        """启用/禁用tracking功能"""
        try:
            self._tracking_enabled = enabled
            # 设置async pose service中的tracking
            if hasattr(self.pose_service, 'processor') and hasattr(self.pose_service.processor, 'set_tracking_enabled'):
                self.pose_service.processor.set_tracking_enabled(enabled)
            # 设置batch processor中的tracking
            if hasattr(self.pose_service, 'processor') and hasattr(self.pose_service.processor, 'pose_estimator'):
                if hasattr(self.pose_service.processor.pose_estimator, 'set_tracking_enabled'):
                    self.pose_service.processor.pose_estimator.set_tracking_enabled(enabled)
            logger.info(f"Tracking {'enabled' if enabled else 'disabled'} in video transform track")
        except Exception as e:
            logger.warning(f"Failed to set tracking enabled: {e}")

    def is_tracking_enabled(self) -> bool:
        """检查tracking是否启用"""
        return self._tracking_enabled

    def stop(self):
        """停止处理服务并清理所有资源"""
        logger.info("Stopping video transform track...")
        
        if hasattr(self.pose_service, 'stop_pipeline'):
            # AsyncPoseService - 异步停止
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务
                    task = loop.create_task(self.pose_service.stop_pipeline())
                    logger.info("Created stop pipeline task")
                else:
                    # 如果事件循环未运行，直接运行
                    loop.run_until_complete(self.pose_service.stop_pipeline())
            except Exception as e:
                logger.error(f"Error stopping AsyncPoseService: {e}")
        elif hasattr(self.pose_service, 'stop'):
            # ThreadedPoseService
            self.pose_service.stop()

        # 清理本地引用
        self.pose_service = None
        self.track = None
        
        # 强制垃圾回收
        import gc
        gc.collect()

        logger.info("Video transform track stopped and resources cleaned")

    def __del__(self):
        """析构时清理资源"""
        try:
            self.stop()
        except:
            pass


# Connection management
pcs = set()

# 全局tracking状态管理
_global_tracking_enabled = True
_active_video_tracks = []


def get_global_tracking_status():
    """获取全局tracking状态"""
    return {
        "enabled": _global_tracking_enabled,
        "active_tracks": len(_active_video_tracks)
    }


def set_global_tracking_enabled(enabled: bool):
    """设置全局tracking状态"""
    global _global_tracking_enabled
    _global_tracking_enabled = enabled
    
    # 更新所有活跃的video tracks
    for video_track in _active_video_tracks:
        if hasattr(video_track, 'set_tracking_enabled'):
            video_track.set_tracking_enabled(enabled)
    
    logger.info(f"Global tracking {'enabled' if enabled else 'disabled'} for {len(_active_video_tracks)} tracks")


def reset_global_tracking():
    """重置所有活跃video tracks的tracking"""
    for video_track in _active_video_tracks:
        if hasattr(video_track, 'reset_tracking'):
            video_track.reset_tracking()
    
    logger.info(f"Global tracking reset for {len(_active_video_tracks)} tracks")


def register_video_track(video_track):
    """注册新的video track"""
    if video_track not in _active_video_tracks:
        _active_video_tracks.append(video_track)
        # 应用全局tracking设置
        if hasattr(video_track, 'set_tracking_enabled'):
            video_track.set_tracking_enabled(_global_tracking_enabled)


def unregister_video_track(video_track):
    """注销video track"""
    if video_track in _active_video_tracks:
        _active_video_tracks.remove(video_track)


def cleanup_pcs():
    """Remove closed connections from the set"""
    closed_pcs = {pc for pc in pcs if pc.connectionState == 'closed'}
    pcs.difference_update(closed_pcs)
    return len(closed_pcs)
