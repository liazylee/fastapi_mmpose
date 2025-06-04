# app/pose_service/onnx_inference.py

import logging
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.pose_service.draw_pose_numba import _draw_circle_numba, _draw_line_numba, point_color, palette, skeleton, \
    link_color, default_skeleton, keypoint_colors

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
        providers = ['CPUExecutionProvider'] if self.device == 'cpu' else ['CUDAExecutionProvider']
        sess = ort.InferenceSession(path_or_bytes=self.onnx_file, providers=providers)
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

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def run_inference_batch(self, batch_imgs: np.ndarray) -> List[np.ndarray]:
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
        outputs = self.sess.run(sess_output, sess_input)
        
        return outputs

    def postprocess(self, outputs: List[np.ndarray], model_input_size: Tuple[int, int],
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

# onnx_file = "/home/stanley/jobs/python/AI/fastapi_mmpose/app/pose_service/configs/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504/end2end.onnx"
# pose_estimator = ONNXPoseEstimator(onnx_file, device='cuda')

# 示例使用方法
# def main():
#     # 初始化ONNX推理器
#
#     # 读取图像
#     image_path = "/home/stanley/jobs/python/AI/mmpose/projects/rtmpose/examples/onnxruntime/human-pose.jpeg"
#     img = cv2.imread(image_path)
#
#     # 执行推理
#     vis_img, keypoints, scores = pose_estimator.inference(img)
#
#     # 保存结果
#     cv2.imwrite("output.jpg", vis_img)
#
#     print(f"关键点形状: {keypoints.shape}")
#     print(f"分数形状: {scores.shape}")
#
#
# if __name__ == "__main__":
#     # main()
#     pass
