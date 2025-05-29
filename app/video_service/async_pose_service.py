import asyncio
import time
import concurrent.futures
from logging import getLogger

import torch
from app.config import BatchProcessingConfig, batch_settings
from app.pose_service.core import pose_service
from app.video_service.batch_GPU_process import BatchPoseProcessor
from app.video_service.manage_frame import FrameBuffer

logger = getLogger(__name__)


class AsyncPoseService:
    """Main service coordinating the entire pipeline"""

    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or batch_settings
        self.frame_buffer = FrameBuffer(self.config)
        self.processor = BatchPoseProcessor(self.config)
        self._shutdown_event = asyncio.Event()

        # Initialize event loop and executor
        self.loop = asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_workers
        )

        # Initialize queues
        self.input_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=self.config.max_queue_size)

        # Save reference to pose service
        self.pose_service = pose_service

        # Pipeline control
        self.running = False
        self.tasks = []

        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'total_latency': 0,
            'batch_efficiency': 0
        }

    async def process_frame_async(self, frame):
        """Process a single frame asynchronously through the pipeline

        Args:
            frame: Input frame (numpy array)

        Returns:
            tuple: (processed_frame, pose_results)
        """
        if not self.running:
            await self.start_pipeline()

        # Add timestamp for latency tracking
        metadata = {
            'timestamp': time.time(),
            'frame_id': self.stats['frames_processed'] + 1
        }

        # Put frame in queue for processing
        await self.input_queue.put((frame, metadata))

        # Get result from output queue
        result_frame, result_metadata = await self.output_queue.get()

        # Update statistics
        latency = time.time() - result_metadata['timestamp']
        self.stats['frames_processed'] += 1
        self.stats['total_latency'] += latency

        # Mark task as done
        self.output_queue.task_done()

        # Return processed frame and pose results
        return result_frame, result_metadata.get('pose_results')

    async def start_pipeline(self):
        """Start the processing pipeline"""
        if self.running:
            return

        self.running = True

        # Create just one task for processing and one for monitoring
        self._processing_task = asyncio.create_task(self._optimized_processing_loop())
        self._monitor_task = asyncio.create_task(self._performance_monitor())

        self.tasks = [self._processing_task, self._monitor_task]

        logger.info("Started optimized processing pipeline")
        logger.info("Async pose processing pipeline started")

    async def stop_pipeline(self):
        """正确停止pipeline"""
        logger.info("Stopping async pose processing pipeline...")

        self.running = False
        self._shutdown_event.set()

        # 取消所有任务
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # 等待任务完成
        if self.tasks:
            try:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error during pipeline shutdown: {e}")

        # Clean up resources
        self.executor.shutdown(wait=False)

        self.tasks.clear()
        logger.info("Async pose processing pipeline stopped")

    async def _performance_monitor(self):
        """改进的性能监控，支持优雅关闭"""
        last_check = time.time()
        last_frame_count = 0

        try:
            while self.running and not self._shutdown_event.is_set():
                try:
                    # 使用wait_for支持中断
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=3.0
                    )
                    break  # 收到关闭信号
                except asyncio.TimeoutError:
                    pass  # 正常的3秒超时，继续监控

                if not self.running:
                    break

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

        except asyncio.CancelledError:
            logger.info("Performance monitor cancelled")
            raise
        except Exception as e:
            logger.error(f"Performance monitor error: {e}")

    async def _optimized_processing_loop(self):
        """Single optimized loop for frame collection, processing and distribution"""
        while self.running:
            # 1. Collect frames into a batch efficiently
            batch_frames = []
            batch_metadata = []

            # Collect frames up to batch size or timeout
            collection_start = time.time()
            while len(batch_frames) < self.config.batch_size:
                try:
                    # Short timeout to balance batch filling vs latency
                    frame, metadata = await asyncio.wait_for(
                        self.input_queue.get(),
                        timeout=self.config.batch_timeout_ms / 1000
                    )
                    batch_frames.append(frame)
                    batch_metadata.append(metadata)
                except asyncio.TimeoutError:
                    # Process whatever we have if we timeout
                    if batch_frames:
                        break
                except Exception as e:
                    logger.error(f"Error collecting frames: {e}")
                    continue

            if not batch_frames:
                # No frames to process, sleep briefly to avoid busy loop
                await asyncio.sleep(0.001)
                continue

            try:
                # 2. Prepare tensors on GPU (single batch preparation)
                batch_size = len(batch_frames)
                height, width = batch_frames[0].shape[:2]

                # Create tensors directly on GPU
                batch_tensor = torch.zeros(
                    (batch_size, 3, height, width),
                    dtype=torch.float32,
                    device=self.config.device
                )

                # Convert frames to tensor in one operation
                for i, frame in enumerate(batch_frames):
                    # Convert numpy to tensor directly on GPU
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(
                        self.config.device, non_blocking=True
                    )
                    batch_tensor[i] = frame_tensor

                # 3. Process batch on GPU (detection + pose estimation)
                det_results, pose_results = await self.loop.run_in_executor(
                    self.executor,
                    self._process_batch_on_gpu,
                    batch_tensor,
                    batch_frames
                )

                # 4. Process results and distribute (all in single loop)
                for i in range(batch_size):
                    frame = batch_frames[i]
                    metadata = batch_metadata[i]

                    if pose_results[i] is not None:
                        # Get visualization result (already GPU-accelerated)
                        vis_frame = pose_results[i][0]  # Assuming pose_results returns (vis_img, pose_data)
                        pose_data = pose_results[i][1]

                        # Add pose results to metadata
                        metadata['pose_results'] = pose_data

                        # Send processed frame to output
                        await self.output_queue.put((vis_frame, metadata))
                    else:
                        # If processing failed, return original frame
                        await self.output_queue.put((frame, metadata))

                    # Explicitly mark task as done
                    self.input_queue.task_done()

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                # Return original frames on error
                for i, (frame, metadata) in enumerate(zip(batch_frames, batch_metadata)):
                    await self.output_queue.put((frame, metadata))
                    self.input_queue.task_done()

        logger.info("Processing loop stopped")

    def _process_batch_on_gpu(self, batch_tensor, original_frames):
        """Process a batch of frames on GPU for detection and pose estimation"""
        batch_size = len(original_frames)
        det_results = []
        pose_results = []

        # Process each frame with GPU acceleration
        for i in range(batch_size):
            # Extract single frame tensor
            frame_tensor = batch_tensor[i]

            # Convert tensor back to numpy for MMPose (until we have true batched inference)
            frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')

            # Process with pose service (detection + pose)
            vis_img, pose_result = self.pose_service.process_image(frame_np)

            det_results.append(None)  # Placeholder
            pose_results.append((vis_img, pose_result))

        return det_results, pose_results