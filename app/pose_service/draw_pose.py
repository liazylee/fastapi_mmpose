"""
@author:liazylee
@license: Apache Licence
@time: 29/05/2025 20:47
@contact: li233111@gmail.com
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import numpy as np
import torch
# Add to imports
from numba import jit, prange

from app.video_service.helper import timeit

keypoint_colors = [
    (255, 0, 0),  # 0: nose
    (255, 85, 0),  # 1: left_eye
    (255, 170, 0),  # 2: right_eye
    (255, 255, 0),  # 3: left_ear
    (170, 255, 0),  # 4: right_ear
    (85, 255, 0),  # 5: left_shoulder
    (0, 255, 0),  # 6: right_shoulder
    (0, 255, 85),  # 7: left_elbow
    (0, 255, 170),  # 8: right_elbow
    (0, 255, 255),  # 9: left_wrist
    (0, 170, 255),  # 10: right_wrist
    (0, 85, 255),  # 11: left_hip
    (0, 0, 255),  # 12: right_hip
    (85, 0, 255),  # 13: left_knee
    (170, 0, 255),  # 14: right_knee
    (255, 0, 255),  # 15: left_ankle
    (255, 0, 170),  # 16: right_ankle
]
default_skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 躯干
    [5, 11], [6, 12], [5, 6],  # 肩膀连接
    [5, 7], [6, 8], [7, 9], [8, 10],  # 手臂
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
    [0, 5], [0, 6]  # 头部到肩膀 (可选)
]
default_coco_skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13],
    [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

COCO_SKELETON = [[x - 1, y - 1] for x, y in default_coco_skeleton]


# Add these Numba-accelerated drawing functions
@jit(nopython=True, parallel=True)
def _draw_circle_numba(img, x, y, radius, color, thickness):
    height, width = img.shape[:2]
    radius_sq = radius ** 2
    inner_sq = (radius - thickness) ** 2

    for i in prange(x - radius, x + radius + 1):
        for j in prange(y - radius, y + radius + 1):
            if 0 <= i < width and 0 <= j < height:
                dist_sq = (i - x) ** 2 + (j - y) ** 2
                if thickness < 0:  # 实心圆
                    if dist_sq <= radius_sq:
                        img[j, i] = color
                else:
                    if inner_sq <= dist_sq <= radius_sq:
                        img[j, i] = color


@jit(nopython=True)
def _draw_line_numba(img, x1, y1, x2, y2, color, thickness=1):
    """Numba-accelerated line drawing using Bresenham's algorithm"""
    height, width = img.shape[:2]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < width and 0 <= y1 < height:
            img[y1, x1, 0] = color[0]
            img[y1, x1, 1] = color[1]
            img[y1, x1, 2] = color[2]

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


# @timeit
def draw_poses_numba(image, pose_results, skeleton=None):
    """Numba-accelerated pose visualization"""
    vis_img = image.copy()

    # Get skeleton information
    if not skeleton:
        skeleton = COCO_SKELETON

    for pose_result in pose_results:
        if hasattr(pose_result, 'pred_instances'):
            pred_instances = pose_result.pred_instances

            if hasattr(pred_instances, 'keypoints') and hasattr(pred_instances, 'keypoint_scores'):
                keypoints_array = pred_instances.keypoints
                scores_array = pred_instances.keypoint_scores

                num_persons = keypoints_array.shape[0] if len(keypoints_array.shape) > 2 else 1

                for person_idx in range(num_persons):
                    if len(keypoints_array.shape) > 2:
                        keypoints = keypoints_array[person_idx]
                        scores = scores_array[person_idx]
                    else:
                        keypoints = keypoints_array
                        scores = scores_array

                    # Draw keypoints with Numba
                    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                        if score > 0.3:  # Confidence threshold
                            x, y = int(kpt[0]), int(kpt[1])

                            if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                                color = keypoint_colors[i % len(keypoint_colors)]
                                _draw_circle_numba(vis_img, x, y, 3, color, -1)
                                _draw_circle_numba(vis_img, x, y, 4, (255, 255, 255), 1)

                    # Draw connections with Numba
                    for connection in skeleton:
                        pt1_idx, pt2_idx = connection

                        if (pt1_idx < len(scores) and pt2_idx < len(scores) and
                                scores[pt1_idx] > 0.3 and scores[pt2_idx] > 0.3):

                            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))

                            if (0 <= pt1[0] < vis_img.shape[1] and 0 <= pt1[1] < vis_img.shape[0] and
                                    0 <= pt2[0] < vis_img.shape[1] and 0 <= pt2[1] < vis_img.shape[0]):
                                _draw_line_numba(vis_img, pt1[0], pt1[1], pt2[0], pt2[1], (0, 255, 0), 1)

    return vis_img


def create_circle_template(radius, color, device):
    diameter = 2 * radius + 1
    grid_y, grid_x = torch.meshgrid(torch.arange(diameter, device=device), torch.arange(diameter, device=device),
                                    indexing='ij')
    dist_sq = (grid_x - radius) ** 2 + (grid_y - radius) ** 2
    mask = dist_sq <= radius ** 2
    circle_tensor = torch.zeros((3, diameter, diameter), device=device)
    circle_tensor[:, mask] = color.view(3, 1)
    return circle_tensor


# GPU批量绘制关键点
def draw_keypoints_gpu(vis_tensor, keypoints, scores, keypoint_colors, radius=3, score_thresh=0.3):
    _, height, width = vis_tensor.shape

    for person_idx in range(keypoints.shape[0]):
        person_kpts = keypoints[person_idx]
        person_scores = scores[person_idx]

        for kpt_idx, (kpt, score) in enumerate(zip(person_kpts, person_scores)):
            if score > score_thresh:
                x, y = int(kpt[0]), int(kpt[1])
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue

                # Use different color for each keypoint
                color = keypoint_colors[kpt_idx % len(keypoint_colors)]

                # Create circle for this specific keypoint
                circle_tensor = create_circle_template(radius, color, vis_tensor.device)

                x1, x2 = max(0, x - radius), min(width, x + radius + 1)
                y1, y2 = max(0, y - radius), min(height, y + radius + 1)

                c_x1, c_x2 = x1 - (x - radius), radius + (x2 - x)
                c_y1, c_y2 = y1 - (y - radius), radius + (y2 - y)

                vis_tensor[:, y1:y2, x1:x2] = circle_tensor[:, c_y1:c_y2, c_x1:c_x2]


# GPU批量绘制骨架连接线
def draw_skeleton_gpu(vis_tensor, keypoints, scores, skeleton, line_color, score_thresh=0.3):
    _, height, width = vis_tensor.shape
    line_color = line_color.to(vis_tensor.dtype)
    for person_idx in range(keypoints.shape[0]):
        person_kpts = keypoints[person_idx]
        person_scores = scores[person_idx]

        for connection in skeleton:
            pt1_idx, pt2_idx = connection

            if (person_scores[pt1_idx] > score_thresh and person_scores[pt2_idx] > score_thresh):
                x1, y1 = int(person_kpts[pt1_idx, 0]), int(person_kpts[pt1_idx, 1])
                x2, y2 = int(person_kpts[pt2_idx, 0]), int(person_kpts[pt2_idx, 1])

                num_points = max(abs(x2 - x1), abs(y2 - y1)) * 2
                t = torch.linspace(0, 1, num_points, device=vis_tensor.device)

                # 插值为 float32，但要转换成索引
                line_x = (x1 * (1 - t) + x2 * t).round().long()
                line_y = (y1 * (1 - t) + y2 * t).round().long()

                valid_mask = (line_x >= 0) & (line_x < width) & (line_y >= 0) & (line_y < height)
                line_x = line_x[valid_mask]
                line_y = line_y[valid_mask]

                # 使用 advanced indexing 赋值颜色（必须 shape = [3, N]）
                for c in range(3):
                    vis_tensor[c, line_y, line_x] = line_color[c]


# 主函数：GPU加速姿态绘制
@timeit
def draw_poses_gpu(image, pose_results, skeleton=None, device='cuda'):
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).to(device).permute(2, 0, 1).float()
    else:
        image_tensor = image.clone()

    vis_tensor = image_tensor.clone()

    skeleton = skeleton or default_skeleton

    keypoint_colors = torch.tensor([
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170]
    ], device=device).float()

    line_color = torch.tensor([0, 255, 0], device=device).float()

    for pose_result in pose_results:
        pred_instances = pose_result.pred_instances

        keypoints = pred_instances.keypoints
        scores = pred_instances.keypoint_scores

        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, device=device)
            scores = torch.tensor(scores, device=device)

        draw_keypoints_gpu(vis_tensor, keypoints, scores, keypoint_colors, radius=3)
        draw_skeleton_gpu(vis_tensor, keypoints, scores, skeleton, line_color)

    vis_img = vis_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    return vis_img
