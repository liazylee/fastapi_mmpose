import time
from concurrent.futures import ThreadPoolExecutor

from app.pose_service.core import pose_service


class DirectGPUProcessor:
    """直接GPU处理，无pipeline延迟"""

    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # 统计
        self.processed_frames = 0
        self.start_time = time.time()

        # GPU预热
        self._warmup()

    def _warmup(self):
        """GPU预热"""
        import numpy as np
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 并行预热
        futures = []
        for i in range(self.num_workers * 3):
            future = self.executor.submit(pose_service.process_image, dummy_frame)
            futures.append(future)

        # 等待预热完成
        for future in futures:
            future.result()

        print(f"DirectGPU warmed up with {self.num_workers} workers")

    def process_frame(self, frame):
        """直接处理帧，无pipeline延迟"""
        try:
            # 直接提交处理任务
            future = self.executor.submit(pose_service.process_image, frame)

            # 立即获取结果（会阻塞直到完成）
            vis_img, pose_results = future.result(timeout=0.1)  # 100ms超时

            self.processed_frames += 1
            if self.processed_frames % 50 == 0:
                elapsed = time.time() - self.start_time
                fps = self.processed_frames / elapsed
                print(f"DirectGPU FPS: {fps:.1f}")

            return vis_img, pose_results

        except Exception as e:
            print(f"DirectGPU processing failed: {e}")
            return frame, None


class FastThreadedProcessor:
    """最简单的多线程处理器"""

    def __init__(self):
        self.processed_frames = 0
        self.start_time = time.time()

        # 预热
        import numpy as np
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            pose_service.process_image(dummy_frame)

        print("FastThreaded processor ready")

    def process_frame(self, frame):
        """最直接的处理"""
        try:
            vis_img, pose_results = pose_service.process_image(frame)

            self.processed_frames += 1
            if self.processed_frames % 50 == 0:
                elapsed = time.time() - self.start_time
                fps = self.processed_frames / elapsed
                print(f"FastThreaded FPS: {fps:.1f}")

            return vis_img, pose_results

        except Exception as e:
            print(f"FastThreaded processing failed: {e}")
            return frame, None
