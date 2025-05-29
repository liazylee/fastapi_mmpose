import gc
import time

import numpy as np
import torch


class GPUStabilizer:
    """稳定GPU性能，减少波动"""

    def __init__(self):
        self.last_cleanup = time.time()
        self.cleanup_interval = 15  # 15秒清理一次（更频繁）
        self.frame_count = 0

    def stabilize_memory(self):
        """定期清理GPU内存"""
        current_time = time.time()

        if current_time - self.last_cleanup > self.cleanup_interval:
            # 清理Python垃圾
            gc.collect()

            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.last_cleanup = current_time
            print(f"GPU memory stabilized - Frame: {self.frame_count}")

    def warmup_streamed_processor(self, streamed_processor, num_warmup=20):
        """预热StreamedGPUProcessor"""
        print("Warming up StreamedGPUProcessor for stable performance...")

        # 创建dummy数据
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # 预热多次，让pipeline充满
        for i in range(num_warmup):
            _ = streamed_processor.process_single_frame(dummy_frame)
            print(f"Warmup {i + 1}/{num_warmup}")

        # 清理预热产生的缓存
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print("StreamedGPUProcessor warmup completed")

    def on_frame_processed(self):
        """每帧调用，用于计数和触发清理"""
        self.frame_count += 1

        # 每100帧检查一次内存
        if self.frame_count % 100 == 0:
            self.stabilize_memory()


# 全局稳定器
stabilizer = GPUStabilizer()
