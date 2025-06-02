import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import batch_settings
from app.video_service.rtlib import rtlib_service

logger = logging.getLogger(__name__)

class BatchProcessor:
    """管理帧批处理以实现高效的GPU处理"""
    
    def __init__(self, config=None):
        self.config = config or batch_settings
        self.batch_size = self.config.batch_size
        self.batch_timeout_ms = self.config.batch_timeout_ms
        self.max_queue_size = self.config.max_queue_size
        
        # 帧队列和结果存储
        self.frame_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.results: Dict[int, Optional[np.ndarray]] = {}
        self.result_events: Dict[int, asyncio.Event] = {}
        
        # 处理任务
        self.processing_task = None
        self.running = False
        
        logger.info(f"批处理器初始化完成: batch_size={self.batch_size}, timeout={self.batch_timeout_ms}ms")
    
    async def start(self):
        """启动批处理任务"""
        if self.processing_task is None:
            self.running = True
            self.processing_task = asyncio.create_task(self._process_batches())
            logger.info("批处理任务已启动")
    
    async def stop(self):
        """停止批处理任务"""
        if self.processing_task:
            self.running = False
            await self.processing_task
            self.processing_task = None
            logger.info("批处理任务已停止")
    
    async def process_frame(self, frame_id: int, image: np.ndarray) -> Optional[np.ndarray]:
        """提交帧进行批处理并等待结果"""
        # 创建结果事件
        event = asyncio.Event()
        self.result_events[frame_id] = event
        
        # 将帧添加到队列
        await self.frame_queue.put((frame_id, image))
        
        # 等待处理完成
        await event.wait()
        
        # 获取并返回结果
        result = self.results.pop(frame_id, None)
        self.result_events.pop(frame_id, None)
        
        return result
    
    async def _process_batches(self):
        """批处理主循环"""
        while self.running:
            batch_frames = []
            batch_ids = []
            
            try:
                # 获取第一帧，超时时间较长以避免空循环
                first_id, first_frame = await asyncio.wait_for(
                    self.frame_queue.get(), 
                    timeout=0.1
                )
                batch_frames.append(first_frame)
                batch_ids.append(first_id)
                self.frame_queue.task_done()
                
                # 收集批次，直到达到批大小或超时
                batch_start_time = time.time()
                timeout_sec = self.batch_timeout_ms / 1000.0
                
                # 尝试填满批次
                while len(batch_frames) < self.batch_size:
                    try:
                        # 计算剩余超时时间
                        elapsed = time.time() - batch_start_time
                        remaining = max(0, timeout_sec - elapsed)
                        
                        if remaining <= 0:
                            break  # 超时，使用当前批次
                        
                        # 等待下一帧
                        frame_id, frame = await asyncio.wait_for(
                            self.frame_queue.get(),
                            timeout=remaining
                        )
                        batch_frames.append(frame)
                        batch_ids.append(frame_id)
                        self.frame_queue.task_done()
                    except asyncio.TimeoutError:
                        break  # 无更多帧，使用当前批次
                
                # 处理当前批次
                if batch_frames:
                    batch_size = len(batch_frames)
                    logger.debug(f"正在处理批次: {batch_size}帧")
                    
                    # 批量处理图像
                    processed_batch = await rtlib_service.predict_vis_batch(batch_frames)
                    
                    # 分发结果并设置事件
                    for i, frame_id in enumerate(batch_ids):
                        self.results[frame_id] = processed_batch[i] if i < len(processed_batch) else None
                        event = self.result_events.get(frame_id)
                        if event:
                            event.set()
            
            except asyncio.TimeoutError:
                # 队列为空，短暂休眠
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"批处理错误: {e}")
                await asyncio.sleep(0.1)  # 避免错误循环

# 全局批处理器实例
batch_processor = BatchProcessor() 