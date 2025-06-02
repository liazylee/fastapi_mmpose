from typing import Optional

import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton


class RTLibService:
    """Service for human pose estimation using RTMLib's Wholebody model."""

    def __init__(self, device: str = 'cpu', backend: str = 'onnxruntime', openpose_skeleton: bool = False):
        """Initialize the RTLibService with the specified device and backend.
        
        Args:
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            backend: Backend for model inference ('onnxruntime', 'opencv', 'openvino')
            openpose_skeleton: Whether to use OpenPose skeleton style
        """
        self.device = device
        self.backend = backend
        self.openpose_skeleton = openpose_skeleton
        self.wholebody = Wholebody(
            mode='balanced',
            to_openpose=openpose_skeleton,
            backend=backend, 
            device=device
        )

    async def predict_vis(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Perform pose estimation and visualization on the input image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Annotated image with pose skeleton or None if no poses detected
        """
        if image is None or image.size == 0:
            return None
            
        return self._process_image(image)

    def _process_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Process an image through the pose estimation pipeline.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Visualization image with pose skeleton drawn
        """
        # Perform pose estimation
        keypoints, scores = self.wholebody(image)
        
        if keypoints is None or len(keypoints) == 0:
            return None
            
        # Draw the poses on the image
        vis_img = draw_skeleton(image, keypoints, scores)
        
        return vis_img


# Global instance with default parameters
rtlib_service = RTLibService()