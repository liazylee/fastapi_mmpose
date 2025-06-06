# app/pose_service/onnx_inference.py

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.pose_service.draw_pose_numba import _draw_circle_numba, _draw_line_numba, point_color, palette, skeleton, \
    link_color, default_skeleton, keypoint_colors
from app.video_service.permance_monitor import time_block

logger = logging.getLogger(__name__)


class ONNXPoseEstimator:
    def __init__(self, onnx_file: str, device: str = 'cuda') -> None:
        """初始化ONNX姿态估计器

        Args:
            onnx_file: ONNX模型文件路径
            device: 推理设备，'cpu'或'cuda'
        """
        self.onnx_file = onnx_file
        self.device = device
        self.sess = self._build_session()
        # 获取输入尺寸
        h, w = self.sess.get_inputs()[0].shape[2:]
        self.model_input_size = (w, h)

    def _build_session(self) -> ort.InferenceSession:
        """构建ONNX运行时会话"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = os.cpu_count()  # 可以根据CPU核心数调整

        providers = ['CPUExecutionProvider'] if self.device == 'cpu' else ['CUDAExecutionProvider']
        sess = ort.InferenceSession(
            path_or_bytes=self.onnx_file,
            providers=providers,
            sess_options=sess_options  # 添加这个参数
        )
        return sess

    def inference(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """对单张图像进行姿态估计

        Args:
            img: 输入图像，BGR格式

        Returns:
            tuple: (可视化图像，关键点，置信度分数)
        """
        # 1. 预处理
        resized_img, center, scale = self.preprocess(img)

        # 2. 推理
        outputs = self.run_inference(resized_img)

        # 3. 后处理
        keypoints, scores = self.postprocess(outputs, self.model_input_size, center, scale)

        # 4. 可视化
        vis_img = self.visualize(img, keypoints, scores)

        return vis_img, keypoints, scores

    async def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """预处理图像"""
        # 获取图像形状
        img_shape = img.shape[:2]
        bbox = np.array([0, 0, img_shape[1], img_shape[0]])

        # 获取中心和缩放
        center, scale = self.bbox_xyxy2cs(bbox, padding=1.25)

        # 进行仿射变换
        resized_img, scale = self.top_down_affine(self.model_input_size, scale, center, img)

        # 归一化图像
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        resized_img = resized_img.astype(np.float32)  # 确保为float32
        resized_img = (resized_img - mean) / std

        return resized_img, center, scale

    def run_inference(self, img: np.ndarray) -> List[np.ndarray]:
        """运行ONNX推理"""
        # 构建输入
        input_tensor = [img.transpose(2, 0, 1)]

        # 构建输出
        sess_input = {self.sess.get_inputs()[0].name: input_tensor}
        sess_output = []
        for out in self.sess.get_outputs():
            sess_output.append(out.name)

        # 运行模型
        outputs = self.sess.run(sess_output, sess_input)

        return outputs

    async def run_inference_batch(self, batch_imgs: np.ndarray) -> List[np.ndarray]:
        """批量运行ONNX推理

        Args:
            batch_imgs: 批量图像，形状为 [N, H, W, C]

        Returns:
            批量输出列表
        """
        # 确保输入数据类型为float32
        batch_imgs = batch_imgs.astype(np.float32)

        # 转换为 [N, C, H, W] 格式
        input_tensor = batch_imgs.transpose(0, 3, 1, 2)

        # 构建输入
        sess_input = {self.sess.get_inputs()[0].name: input_tensor}
        sess_output = []
        for out in self.sess.get_outputs():
            sess_output.append(out.name)

        # 运行模型
        with time_block(name=f'onnx run batch', log_result=True):  # 10083
            outputs = self.sess.run(sess_output, sess_input)

        return outputs

    async def postprocess(self, outputs: List[np.ndarray], model_input_size: Tuple[int, int],
                          center: np.ndarray, scale: np.ndarray, simcc_split_ratio: float = 2.0
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """后处理ONNX模型输出"""
        # 使用simcc解码
        simcc_x, simcc_y = outputs
        keypoints, scores = self.decode(simcc_x, simcc_y, simcc_split_ratio)

        # 重新缩放关键点
        keypoints = keypoints / model_input_size * scale + center - scale / 2

        return keypoints, scores

    def visualize_numba(self, img: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                        thr: float = 0.4) -> np.ndarray:
        """Numba加速的姿态可视化"""
        vis_img = img.copy()

        # 直接处理关键点数组和分数数组
        for kpts, score in zip(keypoints, scores):
            keypoints_num = len(score)

            # 绘制关键点
            for i, (kpt, sc) in enumerate(zip(kpts, score)):
                if sc > thr:
                    x, y = int(kpt[0]), int(kpt[1])
                    if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                        color = keypoint_colors[i % len(keypoint_colors)]
                        _draw_circle_numba(vis_img, x, y, 3, color, -1)

            # 绘制骨架连接
            for connection in default_skeleton:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < keypoints_num and pt2_idx < keypoints_num and
                        score[pt1_idx] > thr and score[pt2_idx] > thr):

                    pt1 = (int(kpts[pt1_idx][0]), int(kpts[pt1_idx][1]))
                    pt2 = (int(kpts[pt2_idx][0]), int(kpts[pt2_idx][1]))

                    if (0 <= pt1[0] < vis_img.shape[1] and 0 <= pt1[1] < vis_img.shape[0] and
                            0 <= pt2[0] < vis_img.shape[1] and 0 <= pt2[1] < vis_img.shape[0]):
                        # 确定连接线颜色
                        color_idx = default_skeleton.index(connection) % len(keypoint_colors)
                        _draw_line_numba(vis_img, pt1[0], pt1[1], pt2[0], pt2[1], keypoint_colors[color_idx], 2)

        return vis_img

    def visualize(self, img: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, thr: float = 0.4,
                  use_numba: bool = False) -> np.ndarray:

        """可视化关键点和骨架"""
        vis_img = img.copy()
        if use_numba:
            vis_img = self.visualize_numba(vis_img, keypoints, scores, thr)
        # default color

        # 绘制关键点和骨架
        for kpts, score in zip(keypoints, scores):
            keypoints_num = len(score)
            for kpt, color in zip(kpts, point_color):
                cv2.circle(vis_img, tuple(kpt.astype(np.int32)), 1, palette[color], 1,
                           cv2.LINE_AA)
            for (u, v), color in zip(skeleton, link_color):
                if u < keypoints_num and v < keypoints_num \
                        and score[u] > thr and score[v] > thr:
                    cv2.line(vis_img, tuple(kpts[u].astype(np.int32)),
                             tuple(kpts[v].astype(np.int32)), palette[color], 1,
                             cv2.LINE_AA)

        return vis_img

    # 以下是从原始代码复制的工具函数
    def bbox_xyxy2cs(self, bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        """将bbox格式从(x,y,w,h)转换为(center, scale)"""
        # 转换单个bbox从(4,)到(1,4)
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        # 获取bbox中心和缩放
        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale

    def _fix_aspect_ratio(self, bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
        """扩展比例以匹配给定的纵横比"""
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def _rotate_point(self, pt: np.ndarray, angle_rad: float) -> np.ndarray:
        """旋转点"""
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """获取第三个点，用于计算仿射矩阵"""
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def get_warp_matrix(self, center: np.ndarray, scale: np.ndarray, rot: float,
                        output_size: Tuple[int, int], shift: Tuple[float, float] = (0., 0.),
                        inv: bool = False) -> np.ndarray:
        """计算仿射变换矩阵"""
        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        # 计算变换矩阵
        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        # 获取源矩形的四个角
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        # 获取目标矩形的四个角
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return warp_mat

    def top_down_affine(self, input_size: Tuple[int, int], bbox_scale: np.ndarray,
                        bbox_center: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """通过仿射变换获取作为模型输入的bbox图像"""
        w, h = input_size
        warp_size = (int(w), int(h))

        # 将bbox重塑为固定的纵横比
        bbox_scale = self._fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

        # 获取仿射矩阵
        center = bbox_center
        scale = bbox_scale
        rot = 0
        warp_mat = self.get_warp_matrix(center, scale, rot, output_size=(w, h))

        # 进行仿射变换
        img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        return img, bbox_scale

    def get_simcc_maximum(self, simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从simcc表示中获取最大响应位置和值"""
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)

        # 获取最大值位置
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)

        # 获取x和y轴之间的最大值
        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        vals = max_val_x
        locs[vals <= 0.] = -1

        # 重塑
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

        return locs, vals

    def decode(self, simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """用高斯调制simcc分布"""
        keypoints, scores = self.get_simcc_maximum(simcc_x, simcc_y)
        keypoints /= simcc_split_ratio

        return keypoints, scores

    # 添加到 onnx_inference.py 中，用于性能分析

    def benchmark_detailed_performance(self):
        """详细的性能基准测试"""
        import torch
        import time
        import GPUtil

        print("🔍 RTMPose ONNX 性能深度分析")
        print("=" * 60)

        # 1. 环境信息
        print(f"ONNX Runtime版本: {ort.__version__}")
        print(f"设备: {self.device}")
        print(f"模型输入尺寸: {self.model_input_size}")

        # GPU信息
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            print(f"GPU: {gpu.name}")
            print(f"GPU内存: {gpu.memoryTotal}MB")
            print(f"GPU驱动: {torch.version.cuda}")

        # 2. 创建测试数据
        dummy_img = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8).astype(np.float32)

        # 归一化
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        dummy_img = (dummy_img - mean) / std

        # 转换格式
        single_input = dummy_img.transpose(2, 0, 1)[np.newaxis, ...]  # [1, C, H, W]

        print(f"输入形状: {single_input.shape}")
        print(f"输入数据类型: {single_input.dtype}")

        # 3. 单帧性能测试
        print("\n📊 单帧性能测试:")
        warmup_runs = 10
        test_runs = 50

        # 预热
        for _ in range(warmup_runs):
            _ = self.sess.run([out.name for out in self.sess.get_outputs()],
                              {self.sess.get_inputs()[0].name: single_input})

        # 测试单帧
        start_time = time.time()
        for _ in range(test_runs):
            _ = self.sess.run([out.name for out in self.sess.get_outputs()],
                              {self.sess.get_inputs()[0].name: single_input})
        single_frame_time = (time.time() - start_time) / test_runs

        print(f"单帧平均时间: {single_frame_time * 1000:.2f}ms")
        print(f"单帧FPS: {1 / single_frame_time:.1f}")

        # 4. 不同batch size测试
        print("\n📊 批处理性能测试:")
        batch_sizes = [1, 2, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            try:
                # 创建batch输入
                batch_input = np.repeat(single_input, batch_size, axis=0)

                # 预热
                for _ in range(5):
                    _ = self.sess.run([out.name for out in self.sess.get_outputs()],
                                      {self.sess.get_inputs()[0].name: batch_input})

                # 测试
                start_time = time.time()
                test_runs_batch = max(1, 20 // batch_size)  # 适应不同batch size
                for _ in range(test_runs_batch):
                    _ = self.sess.run([out.name for out in self.sess.get_outputs()],
                                      {self.sess.get_inputs()[0].name: batch_input})

                batch_time = (time.time() - start_time) / test_runs_batch
                per_frame_time = batch_time / batch_size
                batch_fps = batch_size / batch_time

                print(
                    f"Batch {batch_size:2d}: {batch_time * 1000:6.1f}ms total, {per_frame_time * 1000:5.1f}ms/frame, {batch_fps:5.1f} FPS")

            except Exception as e:
                print(f"Batch {batch_size}: 内存不足 - {e}")
                break

        # 5. 内存使用分析
        print("\n📊 内存使用分析:")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024 ** 2

            # 执行一次大batch推理
            large_batch = np.repeat(single_input, 16, axis=0)
            _ = self.sess.run([out.name for out in self.sess.get_outputs()],
                              {self.sess.get_inputs()[0].name: large_batch})

            memory_after = torch.cuda.memory_allocated() / 1024 ** 2
            memory_peak = torch.cuda.max_memory_allocated() / 1024 ** 2

            print(f"推理前GPU内存: {memory_before:.1f}MB")
            print(f"推理后GPU内存: {memory_after:.1f}MB")
            print(f"内存峰值: {memory_peak:.1f}MB")
            print(f"内存增量: {memory_after - memory_before:.1f}MB")

        # 6. 模型信息分析
        print("\n📊 模型结构分析:")
        print(f"输入节点数: {len(self.sess.get_inputs())}")
        print(f"输出节点数: {len(self.sess.get_outputs())}")

        for i, input_node in enumerate(self.sess.get_inputs()):
            print(f"输入 {i}: {input_node.name}, 形状: {input_node.shape}, 类型: {input_node.type}")

        for i, output_node in enumerate(self.sess.get_outputs()):
            print(f"输出 {i}: {output_node.name}, 形状: {output_node.shape}, 类型: {output_node.type}")

        # 7. 对比分析
        print("\n📊 性能对比分析:")
        expected_single_frame = 8  # RTMPose-M expected time (ms)
        expected_batch_16 = 5  # Expected per-frame time in batch

        single_frame_ms = single_frame_time * 1000
        efficiency_single = expected_single_frame / single_frame_ms

        print(
            f"单帧性能效率: {efficiency_single:.2f}x (期望: {expected_single_frame}ms, 实际: {single_frame_ms:.1f}ms)")

        if single_frame_ms > 20:
            print("⚠️  单帧性能明显偏慢，可能问题:")
            print("   - ONNX模型未充分优化")
            print("   - ONNX Runtime配置问题")
            print("   - 输入数据格式或精度问题")

        print("\n" + "=" * 60)

    # 在 ONNXPoseEstimator 类中添加测试方法
    def run_performance_test(self):
        """运行性能测试"""
        self.benchmark_detailed_performance()

    # 快速添加到现有代码中进行测试
    def quick_performance_check(onnx_file_path, device='cuda'):
        """快速性能检查函数"""
        from app.pose_service.onnx_inference import ONNXPoseEstimator

        pose_estimator = ONNXPoseEstimator(onnx_file_path, device)
        pose_estimator.benchmark_detailed_performance()

        return pose_estimator

# pose_estimator = ONNXPoseEstimator(
#     '/home/stanley/jobs/python/AI/fastapi_mmpose/app/pose_service/configs/rtmpose_onnx/end2end.onnx', 'cuda')
# pose_estimator.benchmark_detailed_performance()
