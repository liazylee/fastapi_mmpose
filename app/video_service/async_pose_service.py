import asyncio
import time
from logging import getLogger

from app.config import BatchProcessingConfig, batch_settings
from app.video_service.batch_GPU_process import BatchPoseProcessor
from app.video_service.manage_frame import FrameBuffer

logger = getLogger(__name__)


class AsyncPoseService:
    """Main service coordinating the entire pipeline"""

    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or batch_settings
        self.frame_buffer = FrameBuffer(self.config)
        self.processor = BatchPoseProcessor(self.config)

        # Pipeline control
        self.running = False
        self.tasks = []

        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'total_latency': 0,
            'batch_efficiency': 0
        }

    async def start_pipeline(self):
        """Start the processing pipeline"""
        self.running = True

        # Start pipeline tasks
        self.tasks = [
            asyncio.create_task(self.frame_buffer.collect_batches()),
            asyncio.create_task(self._batch_processing_loop()),
            asyncio.create_task(self._result_distribution_loop()),
            asyncio.create_task(self._performance_monitor())
        ]

        logger.info("Async pose processing pipeline started")

    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        last_check = time.time()
        last_frame_count = 0

        while self.running:
            await asyncio.sleep(3.0)  # Check every 5 seconds

            current_time = time.time()
            current_frames = self.stats['frames_processed']

            # Calculate metrics
            time_diff = current_time - last_check
            frame_diff = current_frames - last_frame_count

            if time_diff > 0:
                fps = frame_diff / time_diff
                avg_latency = self.stats['total_latency'] / max(current_frames, 1)

                logger.info(
                    f"Batch Processing Stats - FPS: {fps:.1f}, "
                    f"Avg Latency: {avg_latency * 1000:.1f}ms, "
                    f"Total Frames: {current_frames}"
                )

            last_check = current_time
            last_frame_count = current_frames

    async def _result_distribution_loop(self):
        """Distribute results back to requesting frames"""
        while self.running:
            try:
                result_item = await self.frame_buffer.result_queue.get()

                # Extract result components
                batch_item, vis_img, pose_results = result_item['result']

                # Set result in the frame's future
                if not batch_item['future'].done():
                    batch_item['future'].set_result((vis_img, pose_results))

            except Exception as e:
                logger.error(f"Result distribution error: {e}")

    async def _batch_processing_loop(self):
        """Main batch processing loop"""
        while self.running:
            try:
                # Get batch from queue
                batch = await self.frame_buffer.batch_queue.get()

                # Process batch
                start_time = time.time()
                results = await self.processor.process_batch(batch)
                process_time = time.time() - start_time

                # Queue results
                for result in results:
                    await self.frame_buffer.result_queue.put({
                        'result': result,
                        'process_time': process_time / len(results)
                    })

                # Update stats
                self.stats['frames_processed'] += len(batch)
                self.stats['total_latency'] += process_time

            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    async def process_frame_async(self, frame):
        """Public interface for frame processing"""
        if not self.running:
            await self.start_pipeline()

        # Submit frame and get future
        result_future = await self.frame_buffer.add_frame(frame)

        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(result_future, timeout=1)
            return result
        except asyncio.TimeoutError:
            logger.warning("Frame processing timeout")
            return frame, None  # Return original frame
