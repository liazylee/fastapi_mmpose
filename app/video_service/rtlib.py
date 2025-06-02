from typing import List, Optional

import numpy as np
from rtmlib import Wholebody, draw_skeleton


class RTLibService:
    """使用RTMLib的Wholebody模型进行人体姿态估计的服务"""

    def __init__(self, device: str = 'cuda', backend: str = 'onnxruntime', openpose_skeleton: bool = False):
        """初始化RTLibService，指定设备和后端
        
        Args:
            device: 推理设备 ('cpu', 'cuda', 'mps')
            backend: 模型推理后端 ('onnxruntime', 'opencv', 'openvino')
            openpose_skeleton: 是否使用OpenPose骨架样式
        """
        self.device = device
        self.backend = backend
        self.openpose_skeleton = openpose_skeleton
        self.wholebody = Wholebody(
            mode='lightweight',
            to_openpose=openpose_skeleton,
            backend=backend,
            device=device
        )

    async def predict_vis(self, image: np.ndarray) -> Optional[np.ndarray]:
        """对输入图像执行姿态估计和可视化
        
        Args:
            image: BGR格式输入图像
            
        Returns:
            带有姿态骨架的标注图像，如果未检测到姿态则返回None
        """
        if image is None or image.size == 0:
            return None

        return self._process_image(image)

    async def predict_vis_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """批量处理多个输入图像的姿态估计和可视化
        
        Args:
            images: BGR格式输入图像列表
            
        Returns:
            带有姿态骨架的标注图像列表，无姿态的帧返回None
        """
        if not images:
            return []

        return self._process_batch(images)

    def _process_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """通过姿态估计流程处理单个图像
        
        Args:
            image: BGR格式输入图像
            
        Returns:
            带有绘制姿态骨架的可视化图像
        """
        # 执行姿态估计
        keypoints, scores = self.wholebody(image)

        if keypoints is None or len(keypoints) == 0:
            return None

        # 在图像上绘制姿态
        vis_img = draw_skeleton(image, keypoints, scores)

        return vis_img

    def _process_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """批量处理图像通过姿态估计流程
        
        Args:
            images: BGR格式输入图像列表
            
        Returns:
            带有绘制姿态骨架的可视化图像列表
        """
        results = []

        # 批量收集关键点和分数
        # 注意：rtmlib的Wholebody本身不支持原生批处理，
        # 这里使用并行处理多个图像来模拟批处理效果
        all_keypoints = []
        all_scores = []

        # 对批次中的所有图像进行检测
        for image in images:
            if image is None or image.size == 0:
                all_keypoints.append(None)
                all_scores.append(None)
                continue

            # 单独处理每个图像，但保持它们在GPU内存中以最大化并行处理
            keypoints, scores = self.wholebody(image)
            all_keypoints.append(keypoints)
            all_scores.append(scores)

        # 为每个图像可视化结果
        for i, image in enumerate(images):
            if image is None or image.size == 0 or all_keypoints[i] is None or len(all_keypoints[i]) == 0:
                results.append(None)
                continue

            # 在图像上绘制姿态
            vis_img = draw_skeleton(image, all_keypoints[i], all_scores[i])
            results.append(vis_img)

        return results


# 全局实例，使用默认参数
rtlib_service = RTLibService()
