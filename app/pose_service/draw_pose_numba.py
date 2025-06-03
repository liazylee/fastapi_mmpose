import numpy as np
from numba import jit, prange

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
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 躯干
    [5, 11], [6, 12], [5, 6],  # 肩膀连接
    [5, 7], [6, 8], [7, 9], [8, 10],  # 手臂
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
    [0, 5], [0, 6]  # 头部到肩膀 (可选)
]
# Convert keypoint_colors to a numpy array for Numba compatibility
keypoint_colors = np.array([
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
], dtype=np.uint8)


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
    """Draw a line using Bresenham's algorithm with explicit color handling"""
    height, width = img.shape[:2]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    # Extract color components (ensuring Numba compatibility)
    b, g, r = color[0], color[1], color[2]

    while True:
        if 0 <= x1 < width and 0 <= y1 < height:
            img[y1, x1, 0] = b
            img[y1, x1, 1] = g
            img[y1, x1, 2] = r

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
