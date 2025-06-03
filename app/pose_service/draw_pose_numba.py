import cv2
import numpy as np
from numba import jit

skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
            (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
            (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
            (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
            (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
            (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
            (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
            (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
            (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
            (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
            (129, 130), (130, 131), (131, 132)]
palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
           [255, 153, 255], [102, 178, 255], [255, 51, 51]]
link_color = [
    1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
    2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
]
point_color = [
    0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
    4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
    4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
]
default_skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 腿部
    [5, 11], [6, 12], [5, 6],  # 躯干
    [5, 7], [6, 8], [7, 9], [8, 10],  # 手臂
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
    [0, 5], [0, 6]  # 头部到肩膀
]
# Convert keypoint_colors to a numpy array for Numba compatibility
keypoint_colors = np.array([
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
], dtype=np.uint8)

# 统一的颜色配置
LINK_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # 绿色连线
POINT_COLOR = np.array([0, 255, 255], dtype=np.uint8)  # 青色关键点
POINT_BORDER_COLOR = np.array([255, 255, 255], dtype=np.uint8)  # 白色边框
ID_TEXT_COLOR = np.array([255, 255, 255], dtype=np.uint8)  # 白色文字
ID_BG_COLOR = np.array([50, 50, 50], dtype=np.uint8)  # 深灰色背景
ID_BORDER_COLOR = np.array([255, 255, 255], dtype=np.uint8)  # 白色边框


@jit(nopython=True)
def _draw_line_numba(img, x1, y1, x2, y2, color, thickness=1):
    """使用Bresenham算法绘制线条，支持更细的线条"""
    height, width = img.shape[:2]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    # 提取颜色分量
    b, g, r = color[0], color[1], color[2]

    # 主线条
    curr_x, curr_y = x1, y1
    while True:
        # 绘制当前点及其邻域以实现厚度
        for dx_offset in range(-thickness // 2, thickness // 2 + 1):
            for dy_offset in range(-thickness // 2, thickness // 2 + 1):
                px = curr_x + dx_offset
                py = curr_y + dy_offset
                if 0 <= px < width and 0 <= py < height:
                    img[py, px, 0] = b
                    img[py, px, 1] = g
                    img[py, px, 2] = r

        if curr_x == x2 and curr_y == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            curr_x += sx
        if e2 < dx:
            err += dx
            curr_y += sy


@jit(nopython=True)
def _draw_circle_numba(img, x, y, radius, color, thickness=-1):
    """绘制圆形，支持实心和空心"""
    height, width = img.shape[:2]
    radius_sq = radius ** 2

    b, g, r = color[0], color[1], color[2]

    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if 0 <= i < width and 0 <= j < height:
                dist_sq = (i - x) ** 2 + (j - y) ** 2

                if thickness < 0:  # 实心圆
                    if dist_sq <= radius_sq:
                        img[j, i, 0] = b
                        img[j, i, 1] = g
                        img[j, i, 2] = r
                else:  # 空心圆
                    inner_sq = (radius - thickness) ** 2
                    if inner_sq <= dist_sq <= radius_sq:
                        img[j, i, 0] = b
                        img[j, i, 1] = g
                        img[j, i, 2] = r


@jit(nopython=True)
def _draw_keypoints_numba(img, keypoints, scores, threshold=0.3):
    """使用Numba绘制关键点，更细的边框"""
    for i in range(len(keypoints)):
        if scores[i] > threshold:
            x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
            # 绘制关键点 - 更小的半径
            _draw_circle_numba(img, x, y, 2, POINT_COLOR, -1)
            # 绘制白色边框 - 更细的边框
            _draw_circle_numba(img, x, y, 4, POINT_BORDER_COLOR, 1)


@jit(nopython=True)
def _draw_skeleton_numba(img, keypoints, scores, skeleton_connections, threshold=0.3):
    """使用Numba绘制骨架连接，更细的线条"""
    for connection in skeleton_connections:
        kpt_a, kpt_b = connection[0], connection[1]
        if kpt_a < len(keypoints) and kpt_b < len(keypoints):
            if scores[kpt_a] > threshold and scores[kpt_b] > threshold:
                x1, y1 = int(keypoints[kpt_a, 0]), int(keypoints[kpt_a, 1])
                x2, y2 = int(keypoints[kpt_b, 0]), int(keypoints[kpt_b, 1])
                # 更细的线条厚度
                _draw_line_numba(img, x1, y1, x2, y2, LINK_COLOR, thickness=1)


@jit(nopython=True)
def _draw_ellipse_numba(img, center_x, center_y, radius_x, radius_y, start_angle, end_angle, color, thickness=-1):
    """绘制椭圆弧（用于半圆）"""
    height, width = img.shape[:2]
    b, g, r = color[0], color[1], color[2]

    # 简化的椭圆绘制 - 绘制半圆
    for angle in range(start_angle, end_angle + 1):
        rad = np.pi * angle / 180.0
        x = int(center_x + radius_x * np.cos(rad))
        y = int(center_y + radius_y * np.sin(rad))

        if 0 <= x < width and 0 <= y < height:
            if thickness < 0:  # 实心
                # 填充半圆区域
                for r_offset in range(radius_x):
                    inner_x = int(center_x + r_offset * np.cos(rad))
                    inner_y = int(center_y + r_offset * np.sin(rad))
                    if 0 <= inner_x < width and 0 <= inner_y < height:
                        img[inner_y, inner_x, 0] = b
                        img[inner_y, inner_x, 1] = g
                        img[inner_y, inner_x, 2] = r
            else:  # 边框
                img[y, x, 0] = b
                img[y, x, 1] = g
                img[y, x, 2] = r


def draw_style_id_opencv(img, position, track_id):
    """使用OpenCV绘制NBA风格ID - 更细的字体"""
    if position is None:
        return

    x, y = int(position[0]), int(position[1])
    y += 20  # 在脚踝下方

    # 确保ID在图像范围内
    if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
        return

    # NBA风格配置 - 更小更细
    circle_radius = 12  # 更小的半圆半径
    circle_spacing = 24  # 更紧密的间距

    # 计算两个半圆的中心位置
    left_center = (x - circle_spacing // 2, y)
    right_center = (x + circle_spacing // 2, y)

    # 绘制左半圆
    cv2.ellipse(img, left_center, (circle_radius, circle_radius),
                0, 90, 270, (50, 50, 50), -1)  # 填充
    cv2.ellipse(img, left_center, (circle_radius, circle_radius),
                0, 90, 270, (255, 255, 255), 1)  # 更细的边框

    # 绘制右半圆
    cv2.ellipse(img, right_center, (circle_radius, circle_radius),
                0, 270, 90, (50, 50, 50), -1)  # 填充
    cv2.ellipse(img, right_center, (circle_radius, circle_radius),
                0, 270, 90, (255, 255, 255), 1)  # 更细的边框

    # 绘制中间的连接矩形
    rect_top = y - circle_radius
    rect_bottom = y + circle_radius
    rect_left = left_center[0]
    rect_right = right_center[0]

    cv2.rectangle(img, (rect_left, rect_top), (rect_right, rect_bottom),
                  (50, 50, 50), -1)
    cv2.line(img, (rect_left, rect_top), (rect_right, rect_top),
             (255, 255, 255), 1)  # 更细的线条
    cv2.line(img, (rect_left, rect_bottom), (rect_right, rect_bottom),
             (255, 255, 255), 1)  # 更细的线条

    # 绘制ID文本 - 更小更细的字体
    text = str(track_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # 更小的字体
    thickness = 1  # 更细的字体

    # 获取文本大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 计算文本居中位置
    text_x = x - text_width // 2
    text_y = y + text_height // 2

    # 绘制白色ID文字
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


def draw_multi_person_pose_numba(img, tracked_objects, pose_score_thr=0.3):
    """使用Numba绘制多人姿态和tracking ID"""
    vis_img = img.copy()

    # 转换骨架连接为numpy数组以供Numba使用
    skeleton_connections = np.array(default_skeleton, dtype=np.int32)

    for obj in tracked_objects:
        keypoints = obj['keypoints']
        scores = obj['scores']
        track_id = obj['track_id']

        # 确保keypoints是2D数组
        if len(keypoints.shape) == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if len(scores.shape) == 2 and scores.shape[0] == 1:
            scores = scores[0]

        # 转换为numpy数组
        keypoints = np.ascontiguousarray(keypoints.astype(np.float32))
        scores = np.ascontiguousarray(scores.astype(np.float32))

        # 使用Numba绘制骨架连接
        _draw_skeleton_numba(vis_img, keypoints, scores, skeleton_connections, pose_score_thr)

        # 使用Numba绘制关键点
        _draw_keypoints_numba(vis_img, keypoints, scores, pose_score_thr)

        # 获取脚踝位置用于显示ID
        id_position = _get_ankle_position_numba(keypoints, scores)

        # 使用OpenCV绘制NBA风格ID（因为文字渲染复杂度，保留OpenCV）
        draw_style_id_opencv(vis_img, id_position, track_id)

    return vis_img


@jit(nopython=True)
def _get_ankle_position_numba(keypoints, scores):
    """获取脚踝位置用于显示ID - Numba版本"""
    left_ankle_idx = 15  # left_ankle
    right_ankle_idx = 16  # right_ankle
    threshold = 0.3

    # 优先使用左脚踝
    if left_ankle_idx < len(scores) and scores[left_ankle_idx] > threshold:
        return keypoints[left_ankle_idx]
    # 如果左脚踝不可见，使用右脚踝
    elif right_ankle_idx < len(scores) and scores[right_ankle_idx] > threshold:
        return keypoints[right_ankle_idx]
    # 如果都不可见，使用重心位置
    else:
        valid_count = 0
        sum_x, sum_y = 0.0, 0.0

        for i in range(len(scores)):
            if scores[i] > threshold:
                sum_x += keypoints[i, 0]
                sum_y += keypoints[i, 1]
                valid_count += 1

        if valid_count > 0:
            return np.array([sum_x / valid_count, sum_y / valid_count], dtype=np.float32)
        else:
            return None
