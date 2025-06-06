import asyncio
import logging
import time
from collections import deque, OrderedDict
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import numpy as np

from app.config import BatchProcessingConfig, batch_settings
from app.video_service.batch_GPU_process_onnx import BatchPoseProcessorONNX

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Frame data container with sequence tracking"""
    frame: np.ndarray
    timestamp: float
    frame_id: str
    sequence_number: int  # Added sequence number for ordering

    @classmethod
    def create(cls, frame: np.ndarray, sequence_number: int) -> 'FrameData':
        import uuid
        return cls(
            frame=frame,
            timestamp=time.time(),
            frame_id=f"seq_{sequence_number:06d}_{str(uuid.uuid4())[:8]}",  # prefix + sequence
            sequence_number=sequence_number
        )


@dataclass
class ProcessedResult:
    """Processed frame result container"""
    frame_id: str
    sequence_number: int  # Added sequence number
    original_frame: np.ndarray
    processed_frame: Optional[np.ndarray]
    pose_results: Optional[Any]
    timestamp: float
    processing_time: float


class FrameCollector:
    """Collects incoming frames into a buffer queue with sequence tracking"""

    def __init__(self, max_buffer_size: int = 30):
        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self.sequence_counter = 0  # Global sequence counter

    def add_frame(self, frame: np.ndarray) -> FrameData:
        """Add frame to buffer with sequence number, return frame data"""
        self.sequence_counter += 1
        frame_data = FrameData.create(frame, self.sequence_counter)
        self.frame_buffer.append(frame_data)
        logger.debug(
            f"Frame {frame_data.frame_id} (seq:{self.sequence_counter}) added to buffer, size: {len(self.frame_buffer)}")
        return frame_data

    def get_batch(self, batch_size: int) -> List[FrameData]:
        """Get a batch of frames from buffer (maintains order)"""
        batch = []
        while len(batch) < batch_size and self.frame_buffer:
            batch.append(self.frame_buffer.popleft())
        return batch

    def get_latest_frame(self) -> Optional[FrameData]:
        """Get the most recent frame"""
        return self.frame_buffer[-1] if self.frame_buffer else None

    def clear_buffer(self):
        """Clear the frame buffer with explicit memory cleanup"""
        # Clear all frame data explicitly
        while self.frame_buffer:
            frame_data = self.frame_buffer.popleft()
            # Explicitly delete numpy array to free memory
            if hasattr(frame_data, 'frame'):
                frame_data.frame = None
            del frame_data

        self.frame_buffer.clear()

        # Force garbage collection
        import gc
        gc.collect()

        logger.info(f"Frame buffer cleared with memory cleanup (was at sequence {self.sequence_counter})")

    def get_buffer_info(self) -> dict:
        """Get buffer status information"""
        return {
            'buffer_size': len(self.frame_buffer),
            'max_size': self.max_buffer_size,
            'utilization': len(self.frame_buffer) / self.max_buffer_size,
            'current_sequence': self.sequence_counter
        }


class BatchProcessor:
    """Handles batch processing of frames asynchronously"""

    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or batch_settings
        self.processor = BatchPoseProcessorONNX(self.config)
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None

        # Performance metrics
        self.total_frames_processed = 0
        self.total_processing_time = 0.0

    async def start_processing(self, frame_collector: FrameCollector, result_manager: 'ResultManager'):
        """Start the batch processing loop"""
        self.is_running = True
        self.processing_task = asyncio.create_task(
            self._processing_loop(frame_collector, result_manager)
        )
        logger.info("Batch processor started")

    async def stop_processing(self):
        """Stop the batch processing with comprehensive cleanup"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            await asyncio.gather(self.processing_task, return_exceptions=True)

        # Clear metrics
        self.total_frames_processed = 0
        self.total_processing_time = 0.0

        logger.info("Batch processor stopped with memory cleanup")

    async def _processing_loop(self, frame_collector: FrameCollector, result_manager: 'ResultManager'):
        """Main processing loop"""
        while self.is_running:
            # Get a batch of frames
            batch_frames = frame_collector.get_batch(self.config.batch_size)

            if not batch_frames:
                # No frames available, sleep briefly
                await asyncio.sleep(0.005)  # 5ms
                continue

            # Process the batch
            start_time = time.time()
            processed_results = await self._process_batch(batch_frames)
            processing_time = time.time() - start_time

            # Update metrics
            self.total_frames_processed += len(batch_frames)
            self.total_processing_time += processing_time

            # Store results
            for result in processed_results:
                result_manager.add_result(result)

            # Log performance
            avg_time_per_frame = processing_time / len(batch_frames) * 1000
            logger.info(f"🚀 Processed batch: {len(batch_frames)} frames in {processing_time * 1000:.1f}ms "
                        f"({avg_time_per_frame:.1f}ms/frame)")

    async def _process_batch(self, batch_frames: List[FrameData]) -> List[ProcessedResult]:
        """Process a batch of frames"""
        # Convert to the format expected by the processor
        batch_items = []
        for frame_data in batch_frames:
            batch_items.append({
                'frame': frame_data.frame,
                'timestamp': frame_data.timestamp,
                'frame_id': frame_data.frame_id,
                'sequence_number': frame_data.sequence_number,  # Add sequence info
                'future': None  # Not used in this architecture
            })

        # Process batch
        start_time = time.time()
        results = await self.processor.process_batch_sync(batch_items)
        processing_time = time.time() - start_time

        # Convert results back to our format
        processed_results = []
        for (batch_item, vis_img, pose_results) in results:
            result = ProcessedResult(
                frame_id=batch_item['frame_id'],
                sequence_number=batch_item['sequence_number'],  # Preserve sequence
                original_frame=batch_item['frame'],
                processed_frame=vis_img,
                pose_results=pose_results,
                timestamp=time.time(),
                processing_time=processing_time / len(results)
            )
            processed_results.append(result)

        return processed_results

    def get_performance_stats(self) -> dict:
        """Get processing performance statistics"""
        if self.total_frames_processed > 0:
            avg_processing_time = self.total_processing_time / self.total_frames_processed
            fps = self.total_frames_processed / self.total_processing_time
        else:
            avg_processing_time = 0
            fps = 0

        return {
            'total_frames': self.total_frames_processed,
            'total_time': self.total_processing_time,
            'avg_time_per_frame': avg_processing_time,
            'processing_fps': fps
        }


class ResultManager:
    """Manages processed results and provides frames in sequence order"""

    def __init__(self, max_results: int = 20):
        self.results = deque(maxlen=max_results)
        self.latest_result: Optional[ProcessedResult] = None
        self.frame_id_to_result = {}  # For quick lookup
        self.adaptive_gap = True  # 启用自适应间隔
        # Sequence-based result management
        self.ordered_results = OrderedDict()  # sequence_number -> ProcessedResult
        self.output_sequence_number = 0  # Next sequence number to output
        self.max_sequence_gap = 30  # Maximum gap to wait for missing frames

    def add_result(self, result: ProcessedResult):
        """Add a processed result with aggressive memory management"""
        if result is None:
            logger.warning("Attempted to add None result")
            return

        self.results.append(result)
        self.latest_result = result
        self.frame_id_to_result[result.frame_id] = result

        # Add to ordered results for sequence-based output
        self.ordered_results[result.sequence_number] = result

        # More aggressive cleanup to prevent memory buildup
        # Clean up old mappings more frequently (keep smaller cache)
        if len(self.frame_id_to_result) > 20:  # Reduced from 50
            oldest_keys = list(self.frame_id_to_result.keys())[:-10]  # Keep only 10
            for key in oldest_keys:
                if key in self.frame_id_to_result:
                    # Explicitly clean up the result before removing
                    old_result = self.frame_id_to_result[key]
                    if hasattr(old_result, 'original_frame'):
                        old_result.original_frame = None
                    if hasattr(old_result, 'processed_frame'):
                        old_result.processed_frame = None
                    if hasattr(old_result, 'pose_results'):
                        old_result.pose_results = None
                    del self.frame_id_to_result[key]

        # Clean up old ordered results more frequently (keep smaller cache)
        if len(self.ordered_results) > 25:  # Reduced from 50
            oldest_keys = list(self.ordered_results.keys())[:-15]  # Keep only 15
            for key in oldest_keys:
                if key in self.ordered_results:
                    # Explicitly clean up the result before removing
                    old_result = self.ordered_results[key]
                    if hasattr(old_result, 'original_frame'):
                        old_result.original_frame = None
                    if hasattr(old_result, 'processed_frame'):
                        old_result.processed_frame = None
                    if hasattr(old_result, 'pose_results'):
                        old_result.pose_results = None
                    del self.ordered_results[key]

        # Note: Garbage collection is only done during stop() to avoid performance impact

    def get_latest_result(self) -> Optional[ProcessedResult]:
        """Get the most recent processed result"""
        return self.latest_result

    def get_result_by_id(self, frame_id: str) -> Optional[ProcessedResult]:
        """Get result by frame ID"""
        return self.frame_id_to_result.get(frame_id)

    def get_next_sequential_frame(self) -> Tuple[Optional[np.ndarray], Optional[Any], bool]:
        """
        Get the next frame in sequence order
        Returns: (frame, pose_results, has_frame)
        """
        # Look for the next expected sequence number
        next_seq = self.output_sequence_number + 1

        if next_seq in self.ordered_results:
            result = self.ordered_results[next_seq]
            self.output_sequence_number = next_seq

            processed_frame = result.processed_frame
            original_frame = result.original_frame

            output_frame = processed_frame if processed_frame is not None else original_frame
            logger.debug(f"Outputting frame in sequence: {result.frame_id} (seq:{next_seq})")
            return (output_frame, result.pose_results, True)

        # 改进的跳帧逻辑
        available_sequences = [seq for seq in self.ordered_results.keys() if seq > next_seq]
        if available_sequences:
            min_available = min(available_sequences)
            gap = min_available - next_seq

            # 动态调整gap阈值
            current_gap_threshold = self.max_sequence_gap
            if self.adaptive_gap:
                # 如果有很多可用帧，可以容忍更大的gap
                if len(available_sequences) > 10:
                    current_gap_threshold = self.max_sequence_gap * 2

            if gap <= current_gap_threshold:
                # 等待缺失帧
                logger.debug(f"Waiting for sequence {next_seq}, next available: {min_available}, gap: {gap}")
                return (None, None, False)
            else:
                # 跳帧，但记录详细信息
                logger.info(
                    f"Adaptive skip: frames {next_seq} to {min_available - 1}, gap: {gap}, available_count: {len(available_sequences)}")
                self.output_sequence_number = min_available - 1
                return self.get_next_sequential_frame()

        return (None, None, False)

    def get_best_available_frame(self) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """Get the best available processed frame for output (fallback method)"""
        if self.latest_result:
            # Fix NumPy array boolean evaluation issue
            processed_frame = self.latest_result.processed_frame
            original_frame = self.latest_result.original_frame

            # Choose processed frame if available, otherwise use original
            if processed_frame is not None:
                output_frame = processed_frame
            else:
                output_frame = original_frame

            return (output_frame, self.latest_result.pose_results)
        return None, None

    def clear_results(self):
        """Clear all results with explicit memory cleanup"""
        # Clear results with explicit numpy array cleanup
        while self.results:
            result = self.results.popleft()
            if hasattr(result, 'original_frame'):
                result.original_frame = None
            if hasattr(result, 'processed_frame'):
                result.processed_frame = None
            if hasattr(result, 'pose_results'):
                result.pose_results = None
            del result

        # Clear ordered results with explicit cleanup
        for seq_num, result in self.ordered_results.items():
            if hasattr(result, 'original_frame'):
                result.original_frame = None
            if hasattr(result, 'processed_frame'):
                result.processed_frame = None
            if hasattr(result, 'pose_results'):
                result.pose_results = None

        self.results.clear()
        self.latest_result = None
        self.frame_id_to_result.clear()
        self.ordered_results.clear()
        self.output_sequence_number = 0

        # Force garbage collection
        import gc
        gc.collect()

        logger.info("Result manager cleared with comprehensive memory cleanup")


class StreamOutputController:
    """Controls the output stream timing and frame delivery"""

    def __init__(self, target_fps: int = 10):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_output_time = 0.0
        self.output_frame_count = 0
        self.adaptive_timing = True  # 启用自适应时序

    def should_output_frame(self) -> bool:
        """改进的输出控制，考虑缓冲区状态"""
        current_time = time.time()
        elapsed = current_time - self.last_output_time

        # 基本时间间隔检查
        if elapsed >= self.frame_interval:
            return True

        # 如果时间还没到，但如果缓冲区有很多帧，可以提前输出
        if self.adaptive_timing and elapsed >= self.frame_interval * 0.8:
            return True

        return False

    def mark_frame_output(self):
        """Mark that a frame has been output"""
        self.last_output_time = time.time()
        self.output_frame_count += 1

    def get_output_stats(self) -> dict:
        """Get output stream statistics"""
        elapsed_total = time.time() - (self.last_output_time - (self.output_frame_count * self.frame_interval))
        actual_fps = self.output_frame_count / elapsed_total if elapsed_total > 0 else 0

        return {
            'target_fps': self.target_fps,
            'actual_fps': actual_fps,
            'total_output_frames': self.output_frame_count
        }


class StreamingPoseProcessor:
    """Main streaming processor that coordinates all components"""

    def __init__(self, config: BatchProcessingConfig = None, target_fps: int = 30):
        self.config = config or batch_settings

        # Core components with reduced buffer sizes to prevent memory buildup
        self.frame_collector = FrameCollector(max_buffer_size=60)  # Reduced from 30
        self.batch_processor = BatchProcessor(self.config)
        self.result_manager = ResultManager(max_results=30)  # Reduced from 20
        self.output_controller = StreamOutputController(target_fps)

        # State management
        self.is_running = False
        self.start_time = time.time()

        logger.info(f"StreamingPoseProcessor initialized - target FPS: {target_fps}, "
                    f"batch size: {self.config.batch_size}, reduced buffer sizes for memory efficiency")

    async def start(self):
        """Start the streaming processor"""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()

        # Start batch processing
        await self.batch_processor.start_processing(self.frame_collector, self.result_manager)

        logger.info("🎬 StreamingPoseProcessor started")

    async def stop(self):
        """Stop the streaming processor with comprehensive memory cleanup"""
        if not self.is_running:
            return

        self.is_running = False

        logger.info("🛑 Starting comprehensive streaming processor cleanup...")

        # 1. Stop batch processing first
        await self.batch_processor.stop_processing()

        # 2. Clear all buffers with explicit memory cleanup
        self.frame_collector.clear_buffer()
        self.result_manager.clear_results()

        # 3. Reset sequence counters
        self.frame_collector.sequence_counter = 0
        self.output_controller.output_frame_count = 0
        self.output_controller.last_output_time = 0.0

        # 4. Clear processor references
        if hasattr(self.batch_processor, 'processor'):
            processor = self.batch_processor.processor
            # Reset any tracking state in the underlying processor
            if hasattr(processor, 'reset_tracking'):
                processor.reset_tracking()
            if hasattr(processor, 'pose_estimator') and hasattr(processor.pose_estimator, 'reset_tracking'):
                processor.pose_estimator.reset_tracking()

        # 5. GPU memory cleanup
        await self._cleanup_gpu_memory()

        # 6. Force garbage collection
        import gc
        gc.collect()

        logger.info("🛑 StreamingPoseProcessor stopped with comprehensive memory cleanup")

    async def _cleanup_gpu_memory(self):
        """Clean up GPU memory if available"""
        try:
            import torch
            if torch.cuda.is_available():
                # Get memory before cleanup
                current_device = torch.cuda.current_device()
                memory_before = torch.cuda.memory_allocated(current_device) / 1024 ** 2  # MB

                # Clear GPU cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Get memory after cleanup
                memory_after = torch.cuda.memory_allocated(current_device) / 1024 ** 2  # MB
                freed_memory = memory_before - memory_after

                if freed_memory > 1.0:  # Only log if significant memory was freed
                    logger.info(f"GPU memory cleanup - Before: {memory_before:.1f}MB, "
                                f"After: {memory_after:.1f}MB, Freed: {freed_memory:.1f}MB")

        except Exception as e:
            logger.debug(f"GPU memory cleanup skipped: {e}")

    def add_frame(self, frame: np.ndarray) -> FrameData:
        """Add a new frame to the processing pipeline (non-blocking)"""
        return self.frame_collector.add_frame(frame)

    def get_output_frame(self) -> Tuple[Optional[np.ndarray], bool]:
        """
        Get frame for output in sequence order with timing control
        Returns: (frame, should_output)
        """
        if not self.output_controller.should_output_frame():
            return None, False

        try:
            # Try to get the next frame in sequence
            frame, pose_results, has_frame = self.result_manager.get_next_sequential_frame()

            if has_frame and frame is not None:
                self.output_controller.mark_frame_output()
                return frame, True

            return None, False

        except Exception as e:
            logger.error(f"Error getting output frame: {e}")
            return None, False

    async def get_output_frame_async(self, timeout_ms: int = 50) -> Tuple[Optional[np.ndarray], bool]:
        """异步获取输出帧，带超时等待"""
        if not self.output_controller.should_output_frame():
            return None, False

        # 尝试多次获取帧，添加小延迟等待处理完成
        max_attempts = 5
        wait_time = timeout_ms / max_attempts / 1000  # 转换为秒

        for attempt in range(max_attempts):
            try:
                frame, pose_results, has_frame = self.result_manager.get_next_sequential_frame()

                if has_frame and frame is not None:
                    self.output_controller.mark_frame_output()
                    return frame, True

                # 如果没有帧，等待一小段时间再试
                if attempt < max_attempts - 1:
                    await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error getting output frame: {e}")
                break

        return None, False

    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive system statistics"""
        uptime = time.time() - self.start_time

        return {
            'uptime_seconds': uptime,
            'frame_collector': self.frame_collector.get_buffer_info(),
            'batch_processor': self.batch_processor.get_performance_stats(),
            'output_controller': self.output_controller.get_output_stats(),
            'result_manager': {
                'total_results': len(self.result_manager.results),
                'ordered_results_count': len(self.result_manager.ordered_results),
                'frame_id_cache_count': len(self.result_manager.frame_id_to_result),
                'output_sequence_number': self.result_manager.output_sequence_number,
                'has_latest': self.result_manager.latest_result is not None
            }
        }
