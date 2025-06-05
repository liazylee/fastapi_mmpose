import asyncio
import gc
import logging
import time
import uuid

from app.config import batch_settings

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Simple frame buffer that aggressively batches frames"""

    def __init__(self, config=batch_settings):
        self.config = config
        self.input_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.result_queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        logger.info(f"FrameBuffer initialized with batch_size: {config.batch_size}, "
                   f"batch_timeout_ms: {config.batch_timeout_ms}")

    async def add_frame(self, frame_data):
        """Add frame with metadata (timestamp, frame_id)"""
        frame_item = {
            'frame': frame_data,
            'timestamp': time.time(),
            'frame_id': uuid.uuid4(),
            'future': asyncio.Future()
        }
        await self.input_queue.put(frame_item)
        logger.debug(f"Frame added to input_queue, current queue size: {self.input_queue.qsize()}")
        return frame_item['future']

    async def collect_batches(self):
        """Aggressive batch collection - grabs as many frames as possible quickly"""
        logger.info(f"Starting aggressive batch collection with batch_size: {self.config.batch_size}")
        
        while True:
            current_batch = []
            
            # Get the first frame (blocking)
            first_frame = await self.input_queue.get()
            current_batch.append(first_frame)
            start_time = time.time()
            
            # Quickly grab more frames without waiting
            while len(current_batch) < self.config.batch_size:
                try:
                    # Very short timeout to grab frames quickly
                    frame_item = await asyncio.wait_for(self.input_queue.get(), timeout=0.001)
                    current_batch.append(frame_item)
                except asyncio.TimeoutError:
                    # No more frames available immediately
                    break
            
            # Check timeout condition
            elapsed = time.time() - start_time
            timeout_reached = elapsed >= (self.config.batch_timeout_ms / 1000)
            
            # Submit batch if we have frames
            if current_batch:
                batch_size = len(current_batch)
                await self.batch_queue.put(current_batch)
                
                if batch_size >= self.config.batch_size:
                    logger.info(f"🎯 FULL BATCH: {batch_size} frames")
                else:
                    logger.info(f"⚡ QUICK BATCH: {batch_size} frames (grabbed in {elapsed*1000:.1f}ms)")

    async def cleanup_all_queues(self):
        """Clean up all queues and resources"""
        logger.info("Starting queue cleanup...")
        
        # Clean queues
        cleaned_input = 0
        while not self.input_queue.empty():
            frame_item = self.input_queue.get_nowait()
            if not frame_item['future'].done():
                frame_item['future'].cancel()
            frame_item['frame'] = None
            cleaned_input += 1
        
        cleaned_batch = 0
        while not self.batch_queue.empty():
            batch = self.batch_queue.get_nowait()
            for frame_item in batch:
                if not frame_item['future'].done():
                    frame_item['future'].cancel()
                frame_item['frame'] = None
                cleaned_batch += 1
        
        cleaned_result = 0
        while not self.result_queue.empty():
            result_item = self.result_queue.get_nowait()
            cleaned_result += 1
        
        logger.info(f"Cleanup completed - {cleaned_input} input, {cleaned_batch} batch, {cleaned_result} result items")
        gc.collect()

    def get_queue_stats(self):
        """Get queue statistics"""
        return {
            'input_queue_size': self.input_queue.qsize(),
            'batch_queue_size': self.batch_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }
