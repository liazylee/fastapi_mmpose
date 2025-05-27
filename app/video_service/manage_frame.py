import asyncio
import time
import uuid

from app.config import batch_settings


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

    async def _submit_batch(self):
        """Submit collected batch for processing"""
        batch = self.current_batch.copy()
        self.current_batch = []
        await self.batch_queue.put(batch)
