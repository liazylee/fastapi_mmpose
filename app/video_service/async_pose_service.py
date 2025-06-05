import asyncio
import time
from logging import getLogger

from app.config import BatchProcessingConfig, batch_settings
from app.video_service.batch_GPU_process_onnx import BatchPoseProcessorONNX
from app.video_service.manage_frame import FrameBuffer

logger = getLogger(__name__)


class AsyncPoseService:
    """Main service coordinating the entire pipeline"""

    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or batch_settings
        self.frame_buffer = FrameBuffer(self.config)
        self.processor = BatchPoseProcessorONNX(self.config)
        self._shutdown_event = asyncio.Event()
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

    async def stop_pipeline(self):
        """正确停止pipeline并清理所有内存"""
        logger.info("Stopping async pose processing pipeline...")

        self.running = False
        self._shutdown_event.set()

        # 1. 取消所有任务
        cancelled_tasks = 0
        for task in self.tasks:
            if not task.done():
                task.cancel()
                cancelled_tasks += 1

        # 2. 等待任务完成
        if self.tasks:
            try:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error during pipeline shutdown: {e}")

        logger.info(f"Cancelled {cancelled_tasks} pipeline tasks")

        # 3. 清理所有队列中的未处理frames
        if hasattr(self.frame_buffer, 'cleanup_all_queues'):
            await self.frame_buffer.cleanup_all_queues()

        # 4. 重置tracking历史数据
        if hasattr(self.processor, 'reset_tracking'):
            self.processor.reset_tracking()

        # 5. 清理多人姿态估计器的tracking历史
        if hasattr(self.processor, 'pose_estimator') and hasattr(self.processor.pose_estimator, 'reset_tracking'):
            self.processor.pose_estimator.reset_tracking()

        # 6. 清理GPU内存
        await self._cleanup_gpu_memory()

        # 7. 重置统计信息
        self.stats = {
            'frames_processed': 0,
            'total_latency': 0,
            'batch_efficiency': 0
        }

        self.tasks.clear()
        logger.info("Async pose processing pipeline stopped and memory cleaned")

    async def _cleanup_gpu_memory(self):
        """清理GPU内存"""
        try:
            import gc
            import torch

            # 强制垃圾回收
            gc.collect()

            # 清理CUDA缓存
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                memory_before = torch.cuda.memory_allocated(current_device) / 1024 ** 2  # MB

                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                memory_after = torch.cuda.memory_allocated(current_device) / 1024 ** 2  # MB
                freed_memory = memory_before - memory_after

                logger.info(
                    f"GPU memory cleanup - Before: {memory_before:.1f}MB, After: {memory_after:.1f}MB, Freed: {freed_memory:.1f}MB")

        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")

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
                print(f'Processing batch len: {len(batch)}')
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

    async def process_frames_concurrent(self, frames):
        """Process multiple frames concurrently - allows batch formation"""
        if not self.running:
            await self.start_pipeline()

        # Submit all frames concurrently
        futures = []
        for frame in frames:
            result_future = await self.frame_buffer.add_frame(frame)
            futures.append(result_future)

        # Wait for all results
        results = []
        for future in futures:
            try:
                result = await asyncio.wait_for(future, timeout=1)
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning("Frame processing timeout")
                results.append((frames[len(results)], None))
        
        return results


async_pose_service = AsyncPoseService()
