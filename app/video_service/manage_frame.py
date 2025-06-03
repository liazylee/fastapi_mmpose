import asyncio
import time
import uuid
import gc
import logging

from app.config import batch_settings

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe frame buffer with batch formation"""

    def __init__(self, config=batch_settings):
        self.config = config
        self.input_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=4)
        self.result_queue = asyncio.Queue(maxsize=config.max_queue_size)

        # Batch formation
        self.current_batch = []
        self.batch_timer = None

    async def add_frame(self, frame_data):
        """Add frame with metadata (timestamp, frame_id)"""
        frame_item = {
            'frame': frame_data,
            'timestamp': time.time(),
            'frame_id': uuid.uuid4(),
            'future': asyncio.Future()  # For result tracking
        }
        await self.input_queue.put(frame_item)
        return frame_item['future']

    async def collect_batches(self):
        """Continuously form batches from input queue"""
        while True:
            # Try to form batch with timeout
            try:
                frame_item = await asyncio.wait_for(
                    self.input_queue.get(),
                    timeout=self.config.batch_timeout_ms / 1000
                )
                self.current_batch.append(frame_item)

                if len(self.current_batch) >= self.config.batch_size:
                    await self._submit_batch()

            except asyncio.TimeoutError:
                if self.current_batch:  # Submit partial batch
                    await self._submit_batch()
            except asyncio.CancelledError:
                # 处理取消信号，清理当前批次
                await self._cleanup_current_batch()
                raise

    async def _submit_batch(self):
        """Submit collected batch for processing"""
        batch = self.current_batch.copy()
        self.current_batch = []
        await self.batch_queue.put(batch)

    async def _cleanup_current_batch(self):
        """清理当前批次中的未处理frames"""
        if self.current_batch:
            logger.info(f"Cleaning up current batch with {len(self.current_batch)} frames")
            for frame_item in self.current_batch:
                if not frame_item['future'].done():
                    frame_item['future'].cancel()
                # 清理frame数据
                frame_item['frame'] = None
            self.current_batch.clear()

    async def cleanup_all_queues(self):
        """清理所有队列中的未处理frames"""
        logger.info("Starting comprehensive queue cleanup...")
        
        # 记录清理前的统计
        input_count = self.input_queue.qsize()
        batch_count = self.batch_queue.qsize()
        result_count = self.result_queue.qsize()
        
        logger.info(f"Queue sizes before cleanup - Input: {input_count}, Batch: {batch_count}, Result: {result_count}")

        # 1. 清理input_queue
        cleaned_input = 0
        while not self.input_queue.empty():
            try:
                frame_item = self.input_queue.get_nowait()
                if not frame_item['future'].done():
                    frame_item['future'].cancel()
                # 清理frame数据
                frame_item['frame'] = None
                frame_item.clear()
                cleaned_input += 1
            except asyncio.QueueEmpty:
                break

        # 2. 清理batch_queue
        cleaned_batch = 0
        while not self.batch_queue.empty():
            try:
                batch = self.batch_queue.get_nowait()
                for frame_item in batch:
                    if not frame_item['future'].done():
                        frame_item['future'].cancel()
                    # 清理frame数据
                    frame_item['frame'] = None
                    frame_item.clear()
                    cleaned_batch += 1
                batch.clear()
            except asyncio.QueueEmpty:
                break

        # 3. 清理result_queue
        cleaned_result = 0
        while not self.result_queue.empty():
            try:
                result_item = self.result_queue.get_nowait()
                # 清理结果数据
                if 'result' in result_item:
                    result_item['result'] = None
                result_item.clear()
                cleaned_result += 1
            except asyncio.QueueEmpty:
                break

        # 4. 清理当前批次
        await self._cleanup_current_batch()

        logger.info(f"Queue cleanup completed - Cleaned {cleaned_input} input frames, {cleaned_batch} batch frames, {cleaned_result} result items")
        
        # 5. 强制垃圾回收
        gc.collect()
        logger.info("Forced garbage collection completed")

    def get_queue_stats(self):
        """获取队列统计信息"""
        return {
            'input_queue_size': self.input_queue.qsize(),
            'batch_queue_size': self.batch_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'current_batch_size': len(self.current_batch)
        }
