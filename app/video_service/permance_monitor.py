import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameStats:
    receive_time: float = 0.0
    processing_time: float = 0.0
    total_time: float = 0.0
    frame_size: tuple = (0, 0)


class PerformanceMonitor:
    """Monitor frame receive vs processing performance"""

    def __init__(self, window_size=100):
        self.window_size = window_size

        # Performance tracking
        self.receive_times = deque(maxlen=window_size)
        self.process_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)

        # Counters
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()

        # Current frame timing
        self.current_frame_start = None
        self.current_receive_end = None
        self.frame_time_history = deque(maxlen=1000)  # Longer history
        self.slow_frames = 0  # Count of slow frames

    def start_frame(self):
        """Call when starting to receive frame"""
        self.current_frame_start = time.time()

    def frame_received(self):
        """Call when frame receive is complete"""
        self.current_receive_end = time.time()
        if self.current_frame_start:
            receive_time = self.current_receive_end - self.current_frame_start
            self.receive_times.append(receive_time)

    def frame_processed(self):
        """Call when frame processing is complete"""
        if self.current_receive_end:
            process_time = time.time() - self.current_receive_end
            total_time = time.time() - self.current_frame_start

            self.process_times.append(process_time)
            self.total_times.append(total_time)
            self.frame_time_history.append(total_time)

            # Count slow frames (>30ms)
            if total_time > 0.03:  # 30ms threshold
                self.slow_frames += 1

            self.frame_count += 1

            if time.time() - self.last_log_time > 2.0:
                self._log_performance()
                self.last_log_time = time.time()

    def _log_performance(self):
        """Log detailed performance metrics with instantaneous FPS"""
        if not self.receive_times or not self.process_times:
            return

        # Calculate averages
        avg_receive = sum(self.receive_times) / len(self.receive_times) * 1000
        avg_process = sum(self.process_times) / len(self.process_times) * 1000
        avg_total = sum(self.total_times) / len(self.total_times) * 1000

        # Calculate BOTH cumulative and instantaneous FPS
        elapsed_total = time.time() - self.start_time
        cumulative_fps = self.frame_count / elapsed_total if elapsed_total > 0 else 0

        # Instantaneous FPS (based on recent frames)
        time_since_last_log = time.time() - self.last_log_time
        recent_frames = len(self.total_times)  # frames in current window
        instantaneous_fps = recent_frames / time_since_last_log if time_since_last_log > 0 else 0

        # Theoretical max FPS based on processing time
        theoretical_fps = 1000 / avg_total if avg_total > 0 else 0

        receive_pct = (avg_receive / avg_total) * 100 if avg_total > 0 else 0
        process_pct = (avg_process / avg_total) * 100 if avg_total > 0 else 0
        # Calculate variance and percentiles
        if len(self.frame_time_history) > 10:
            frame_times_ms = [t * 1000 for t in self.frame_time_history]
            p50 = np.percentile(frame_times_ms, 50)
            p95 = np.percentile(frame_times_ms, 95)
            p99 = np.percentile(frame_times_ms, 99)
            std_dev = np.std(frame_times_ms)
            slow_frame_pct = (self.slow_frames / self.frame_count) * 100
        else:
            p50 = p95 = p99 = std_dev = slow_frame_pct = 0

        print(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║                    PERFORMANCE MONITOR                    ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Instantaneous FPS: {instantaneous_fps:>6.1f}                           ║
        ║ Cumulative FPS:    {cumulative_fps:>6.1f}                           ║  
        ║ Theoretical FPS:   {theoretical_fps:>6.1f}                           ║
        ║ Total Frames:      {self.frame_count:>6}                           ║
        ║                                                           ║
        ║ PERFORMANCE STABILITY:                                    ║
        ║ Frame Time P50:    {p50:>6.1f}ms                              ║
        ║ Frame Time P95:    {p95:>6.1f}ms                              ║
        ║ Frame Time P99:    {p99:>6.1f}ms                              ║
        ║ Std Deviation:     {std_dev:>6.1f}ms                              ║
        ║ Slow Frames:       {slow_frame_pct:>6.1f}%                               ║
        ║                                                           ║
        ║ TIMING BREAKDOWN:                                         ║
        ║ Frame Receive: {avg_receive:>6.1f}ms ({receive_pct:>4.1f}%)                         ║
        ║ GPU Process:   {avg_process:>6.1f}ms ({process_pct:>4.1f}%)                         ║
        ║ Total Time:    {avg_total:>6.1f}ms                                   ║
        ║                                                           ║
        ║ BOTTLENECK: {'RECEIVE' if receive_pct > 60 else 'PROCESSING' if process_pct > 60 else 'UNSTABLE' if std_dev > 5 else 'BALANCED':>10}                                ║
        ╚═══════════════════════════════════════════════════════════╝
                """)


# Global monitor instance
monitor = PerformanceMonitor()
