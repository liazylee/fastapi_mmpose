import functools
import logging
import statistics
import threading
import time
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("video_service")


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

            if time.time() - self.last_log_time > 5.0:
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

        logger.info(f"""
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


@dataclass
class PerformanceStats:
    """性能统计数据"""
    name: str
    total_time: float = 0.0
    count: int = 0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, elapsed_time: float):
        """更新统计数据"""
        self.total_time += elapsed_time
        self.count += 1
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
        self.avg_time = self.total_time / self.count
        self.recent_times.append(elapsed_time)

    def get_recent_avg(self, window_size: int = 10) -> float:
        """获取最近N次的平均时间"""
        recent = list(self.recent_times)[-window_size:]
        return statistics.mean(recent) if recent else 0.0

    def get_fps(self) -> float:
        """获取FPS（基于平均时间）"""
        return 1.0 / self.avg_time if self.avg_time > 0 else 0.0

    def get_recent_fps(self, window_size: int = 10) -> float:
        """获取最近的FPS"""
        recent_avg = self.get_recent_avg(window_size)
        return 1.0 / recent_avg if recent_avg > 0 else 0.0


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, max_history: int = 1000):
        self.stats: Dict[str, PerformanceStats] = {}
        self.max_history = max_history
        self._lock = threading.Lock()
        self.enabled = True

        # 层级统计（用于嵌套调用）
        self.call_stack: List[str] = []
        self.hierarchy_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def is_enabled(self) -> bool:
        return self.enabled

    def enable(self):
        """启用性能监控"""
        self.enabled = True
        logger.info("Performance monitoring enabled")

    def disable(self):
        """禁用性能监控"""
        self.enabled = False
        logger.info("Performance monitoring disabled")

    def record(self, name: str, elapsed_time: float):
        """记录性能数据"""
        if not self.enabled:
            return

        with self._lock:
            if name not in self.stats:
                self.stats[name] = PerformanceStats(name)
            self.stats[name].update(elapsed_time)

    def get_stats(self, name: str) -> Optional[PerformanceStats]:
        """获取指定名称的统计数据"""
        return self.stats.get(name)

    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """获取所有统计数据"""
        return dict(self.stats)

    def reset(self):
        """重置所有统计数据"""
        with self._lock:
            self.stats.clear()
            self.hierarchy_stats.clear()
            self.call_stack.clear()
        logger.info("Performance statistics reset")

    def print_summary(self, sort_by: str = 'avg_time', top_n: int = 20):
        """打印性能摘要"""
        if not self.stats:
            print("No performance data available")
            return

        # 排序
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda x: getattr(x, sort_by, 0),
            reverse=True
        )[:top_n]

        print("\n" + "=" * 80)
        print(f"PERFORMANCE SUMMARY (Top {top_n}, sorted by {sort_by})")
        print("=" * 80)
        print(
            f"{'Function':<30} {'Count':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10} {'FPS':<8}")
        print("-" * 80)

        for stat in sorted_stats:
            print(f"{stat.name:<30} {stat.count:<8} {stat.total_time:<10.3f} "
                  f"{stat.avg_time * 1000:<10.2f} {stat.min_time * 1000:<10.2f} "
                  f"{stat.max_time * 1000:<10.2f} {stat.get_fps():<8.1f}")

        print("=" * 80)

    def get_bottlenecks(self, threshold_ms: float = 50.0) -> List[PerformanceStats]:
        """获取性能瓶颈（超过阈值的函数）"""
        bottlenecks = []
        for stat in self.stats.values():
            if stat.avg_time * 1000 > threshold_ms:  # 转换为毫秒
                bottlenecks.append(stat)

        return sorted(bottlenecks, key=lambda x: x.avg_time, reverse=True)

    def get_recent_performance(self, window_size: int = 10) -> Dict[str, float]:
        """获取最近的性能数据（FPS）"""
        recent_fps = {}
        for name, stat in self.stats.items():
            recent_fps[name] = stat.get_recent_fps(window_size)
        return recent_fps


# 全局性能监控器实例
perf_monitor = PerformanceMonitor()


def timeit(name: Optional[str] = None, enabled: bool = True, log_threshold_ms: float = None):
    """性能监控装饰器
    
    Args:
        name: 自定义名称，默认使用函数名
        enabled: 是否启用监控
        log_threshold_ms: 超过阈值时记录日志（毫秒）
    """

    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled or not perf_monitor.is_enabled():
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            try:
                # 添加到调用栈
                perf_monitor.call_stack.append(func_name)
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # 从调用栈移除
                if perf_monitor.call_stack and perf_monitor.call_stack[-1] == func_name:
                    perf_monitor.call_stack.pop()

                # 记录性能数据
                perf_monitor.record(func_name, elapsed_time)

                # 记录日志（如果超过阈值）
                if log_threshold_ms and elapsed_time * 1000 > log_threshold_ms:
                    logger.warning(f"{func_name} took {elapsed_time * 1000:.2f}ms (threshold: {log_threshold_ms}ms)")

        return wrapper

    return decorator


@contextmanager
def time_block(name: str, log_result: bool = False):
    """上下文管理器，用于测量代码块执行时间
    
    Args:
        name: 代码块名称
        log_result: 是否记录结果到日志
    """
    if not perf_monitor.is_enabled():
        yield
        return

    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        perf_monitor.record(name, elapsed_time)

        if log_result:
            logger.info(f"{name} took {elapsed_time * 1000:.2f}ms")


class FPSCounter:
    """FPS计数器"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        """更新FPS计数器，返回当前FPS"""
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time

        self.frame_times.append(frame_time)

        if len(self.frame_times) > 1:
            avg_frame_time = statistics.mean(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return 0.0

    def get_avg_fps(self) -> float:
        """获取平均FPS"""
        if len(self.frame_times) > 1:
            avg_frame_time = statistics.mean(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return 0.0


class RealTimeMonitor:
    """实时性能监控器"""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.last_update = time.perf_counter()
        self.fps_counter = FPSCounter()

    def should_update(self) -> bool:
        """是否应该更新显示"""
        current_time = time.perf_counter()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False

    def print_realtime_stats(self, key_functions: List[str] = None):
        """打印实时统计信息"""
        if not self.should_update():
            return

        current_fps = self.fps_counter.tick()

        print(f"\r📊 Real-time Performance | FPS: {current_fps:.1f}", end="")

        if key_functions:
            recent_fps = perf_monitor.get_recent_performance()
            key_stats = []
            for func_name in key_functions:
                if func_name in recent_fps:
                    fps = recent_fps[func_name]
                    key_stats.append(f"{func_name.split('.')[-1]}: {fps:.1f}")

            if key_stats:
                print(f" | {' | '.join(key_stats)}", end="")

        print("    ", end="", flush=True)


# 便捷函数
def enable_monitoring():
    """启用性能监控"""
    perf_monitor.enable()


def disable_monitoring():
    """禁用性能监控"""
    perf_monitor.disable()


def reset_stats():
    """重置统计数据"""
    perf_monitor.reset()


def print_performance_summary(sort_by: str = 'avg_time', top_n: int = 20):
    """打印性能摘要"""
    perf_monitor.print_summary(sort_by, top_n)


def get_bottlenecks(threshold_ms: float = 50.0):
    """获取性能瓶颈"""
    return perf_monitor.get_bottlenecks(threshold_ms)


def save_performance_report(filename: str):
    """保存性能报告到文件"""
    with open(filename, 'w') as f:
        f.write("PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")

        stats = perf_monitor.get_all_stats()
        sorted_stats = sorted(stats.values(), key=lambda x: x.avg_time, reverse=True)

        for stat in sorted_stats:
            f.write(f"Function: {stat.name}\n")
            f.write(f"  Count: {stat.count}\n")
            f.write(f"  Total Time: {stat.total_time:.3f}s\n")
            f.write(f"  Average Time: {stat.avg_time * 1000:.2f}ms\n")
            f.write(f"  Min Time: {stat.min_time * 1000:.2f}ms\n")
            f.write(f"  Max Time: {stat.max_time * 1000:.2f}ms\n")
            f.write(f"  FPS: {stat.get_fps():.1f}\n\n")

        # 瓶颈分析
        bottlenecks = perf_monitor.get_bottlenecks(50.0)
        if bottlenecks:
            f.write("BOTTLENECKS (>50ms average):\n")
            f.write("-" * 30 + "\n")
            for bottleneck in bottlenecks:
                f.write(f"⚠️  {bottleneck.name}: {bottleneck.avg_time * 1000:.2f}ms\n")

    logger.info(f"Performance report saved to {filename}")
