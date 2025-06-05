# app/pose_service/multi_person_onnx.py

import logging
from typing import List, Tuple, Dict

import numpy as np
from mmdet.apis import inference_detector

from app.config import batch_settings
from app.pose_service.draw_pose_numba import draw_multi_person_pose_numba
from app.pose_service.iou_tracker import IoUTracker
from app.pose_service.onnx_inference import ONNXPoseEstimator
from app.pose_service.yolo_detector import DetectorWithTracking
from app.video_service.permance_monitor import timeit, time_block

logger = logging.getLogger(__name__)


class MultiPersonONNXPoseEstimator:
    """多人姿态估计器，结合检测器和ONNX姿态模型，支持tracking"""

    def __init__(self, onnx_file: str, device: str = 'cuda', batch_setting: batch_settings = batch_settings):
        """初始化多人姿态估计器

        Args:
            onnx_file: ONNX模型文件路径
            device: 推理设备
            batch_setting: 批处理设置
        """
        self.device = device
        self.pose_estimator = ONNXPoseEstimator(onnx_file, device)
        self.detector = None  # 将在外部设置
        self.settings = batch_setting or batch_settings

        # 检测和姿态估计的阈值
        self.det_score_thr = self.settings.det_score_thr
        self.pose_score_thr = self.settings.pose_score_thr

        # Tracking相关 - 使用独立的IoU tracker
        self.enable_tracking = True
        self.iou_tracker = IoUTracker(
            max_track_age=self.settings.max_track_age,
            iou_threshold=self.settings.iou_threshold
        )

        # 检测器类型标识
        self.detector_type = self.settings.detector_type
        self.use_yolo_tracking = self.settings.use_yolo_tracking  # 是否使用YOLO内置tracking

    def set_detector(self, detector):
        """设置检测器（支持MMDetection或YOLO）"""
        self.detector = detector

    def reset_tracking(self):
        """重置tracking状态并清理内存"""
        if self.detector_type == 'yolo' and hasattr(self.detector, 'yolo_detector'):
            self.detector.yolo_detector.reset_tracking()

        self.iou_tracker.reset_tracking()
        logger.info("All tracking states reset")

    def set_tracking_enabled(self, enabled: bool = True):
        """启用/禁用tracking"""
        self.enable_tracking = enabled

        if self.detector_type == 'yolo' and hasattr(self.detector, 'set_tracking_enabled'):
            self.detector.set_tracking_enabled(enabled and self.use_yolo_tracking)

        self.iou_tracker.set_tracking_enabled(enabled and not self.use_yolo_tracking)

        logger.info(f"Tracking {'enabled' if enabled else 'disabled'}")

    def inference(self, img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """对图像进行多人姿态估计和tracking

        Args:
            img: 输入图像，BGR格式

        Returns:
            tuple: (可视化图像, 关键点列表, 分数列表)
        """
        # 1. 人体检测（可能包含tracking）
        if self.detector_type == 'yolo' and self.use_yolo_tracking:
            # 使用YOLO内置tracking
            bboxes, yolo_track_ids = self._detect_humans_yolo_with_tracking(img)
        else:
            # 使用普通检测
            bboxes = self._detect_humans(img)
            yolo_track_ids = None

        if len(bboxes) == 0:
            # 即使没有检测，也要更新tracking状态
            if self.enable_tracking and not self.use_yolo_tracking:
                self.iou_tracker.track_persons([])
            return img, [], []

        # 2. 批量姿态估计
        current_detections = self._batch_pose_estimation(img, bboxes)

        # 3. Tracking处理
        if self.use_yolo_tracking and yolo_track_ids:
            # 使用YOLO提供的track ID
            tracked_objects = []
            for detection, track_id in zip(current_detections, yolo_track_ids):
                tracked_objects.append({
                    'track_id': track_id,
                    'bbox': detection['bbox'],
                    'keypoints': detection['keypoints'],
                    'scores': detection['scores'],
                    'age': 0
                })
        else:
            # 使用IoU tracker
            tracked_objects = self.iou_tracker.track_persons(current_detections)

        # 4. 可视化
        vis_img = self._visualize_multi_person_with_tracking(img, tracked_objects)

        # 返回格式保持兼容
        keypoints_list = [obj['keypoints'] for obj in tracked_objects]
        scores_list = [obj['scores'] for obj in tracked_objects]

        return vis_img, keypoints_list, scores_list

    def _batch_pose_estimation(self, img: np.ndarray, bboxes: List[np.ndarray]) -> List[Dict]:
        """批量姿态估计，一次性处理多个人"""
        if len(bboxes) == 0:
            return []

        # 1. 批量裁剪和预处理
        batch_person_imgs = []
        batch_centers = []
        batch_scales = []

        for bbox in bboxes:
            person_img = self._crop_person(img, bbox)
            # 预处理单个图像
            resized_img, center, scale = self.pose_estimator.preprocess(person_img)

            # 确保数据类型为float32
            resized_img = resized_img.astype(np.float32)

            batch_person_imgs.append(resized_img)
            batch_centers.append(center)
            batch_scales.append(scale)

        # 2. 构建batch输入
        batch_input = np.stack(batch_person_imgs, axis=0)  # Shape: [N, H, W, C]

        # 3. 批量ONNX推理
        batch_outputs = self.pose_estimator.run_inference_batch(batch_input)

        # 4. 批量后处理
        current_detections = []
        for i, bbox in enumerate(bboxes):
            # 从batch输出中提取单个结果
            single_outputs = [output[i:i + 1] for output in batch_outputs]  # 保持batch维度

            # 后处理单个结果
            keypoints, scores = self.pose_estimator.postprocess(
                single_outputs,
                self.pose_estimator.model_input_size,
                batch_centers[i],
                batch_scales[i]
            )

            # 转换坐标系
            keypoints = self._transform_keypoints_to_original(keypoints, bbox)

            current_detections.append({
                'bbox': bbox[:4],
                'keypoints': keypoints,
                'scores': scores
            })

        return current_detections

    def _detect_humans_yolo_with_tracking(self, img: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """使用YOLO进行检测和tracking
        
        Returns:
            tuple: (人体边界框列表, track_id列表)
        """
        if not hasattr(self, '_detector_tracker'):
            from app.pose_service.yolo_detector import DetectorWithTracking
            self._detector_tracker = DetectorWithTracking()

        tracks = self._detector_tracker.detect_and_track_single(img)
        bboxes = []
        track_ids = []

        for track in tracks:
            bbox = np.array(track['bbox'] + [track['score']])  # [x1, y1, x2, y2, score]
            bboxes.append(bbox)
            track_ids.append(track['track_id'])

        return bboxes, track_ids

    @timeit(name=f'{__name__}._detect_humans', log_threshold_ms=50)
    def _detect_humans(self, img: np.ndarray) -> List[np.ndarray]:
        """检测图像中的人体（不包含tracking）

        Returns:
            人体边界框列表 [[x1, y1, x2, y2, score], ...]
        """
        if self.detector is None:
            logger.warning("检测器未设置，跳过人体检测")
            # 返回整张图作为默认bbox
            h, w = img.shape[:2]
            return [np.array([0, 0, w, h, 1.0])]

        if self.detector_type == 'yolo':
            # 使用YOLO检测器（不带tracking）
            if not hasattr(self, '_yolo_detector'):
                from app.pose_service.yolo_detector import YOLODetector
                self._yolo_detector = YOLODetector()

            detections = self._yolo_detector.detect_single(img)
            # 转换为兼容格式 [x1, y1, x2, y2, score]
            bboxes = []
            for detection in detections:
                bbox = np.array(detection['bbox'] + [detection['score']])
                if bbox[4] > self.det_score_thr:  # 检查置信度阈值
                    bboxes.append(bbox)
            return bboxes
        else:
            # 使用MMDetection进行检测
            result = inference_detector(self.detector, img)
            return self._extract_bboxes_from_mmdet_result(result)

    def _extract_bboxes_from_mmdet_result(self, result) -> List[np.ndarray]:
        """从MMDetection结果中提取边界框"""
        if hasattr(result, 'pred_instances'):
            instances = result.pred_instances.cpu().numpy()
            person_bboxes = []
            for i in range(len(instances.bboxes)):
                bbox = instances.bboxes[i]
                score = instances.scores[i]
                if score > self.det_score_thr:
                    person_bboxes.append(np.append(bbox, score))
            return person_bboxes
        return []

    def _crop_person(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """根据边界框裁剪人体区域

        Args:
            img: 原始图像
            bbox: 边界框 [x1, y1, x2, y2, score]

        Returns:
            裁剪后的人体图像
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)

        # 添加边界检查
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 扩展边界框以包含更多上下文
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 扩展20%
        new_w = int(bbox_w * 1.2)
        new_h = int(bbox_h * 1.2)

        new_x1 = max(0, cx - new_w // 2)
        new_y1 = max(0, cy - new_h // 2)
        new_x2 = min(w, cx + new_w // 2)
        new_y2 = min(h, cy + new_h // 2)

        return img[new_y1:new_y2, new_x1:new_x2]

    def _transform_keypoints_to_original(self, keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """将关键点坐标从裁剪图像坐标系转换回原图坐标系

        Args:
            keypoints: 裁剪图像中的关键点坐标
            bbox: 原始边界框

        Returns:
            原图坐标系中的关键点
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)

        # 计算扩展后的边界框
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        new_w = int(bbox_w * 1.2)
        new_h = int(bbox_h * 1.2)

        new_x1 = cx - new_w // 2
        new_y1 = cy - new_h // 2

        # 转换坐标
        transformed_keypoints = keypoints.copy()
        if len(transformed_keypoints.shape) == 3:
            # 批量处理
            transformed_keypoints[:, :, 0] += new_x1
            transformed_keypoints[:, :, 1] += new_y1
        else:
            # 单个关键点集
            transformed_keypoints[:, 0] += new_x1
            transformed_keypoints[:, 1] += new_y1

        return transformed_keypoints

    def _visualize_multi_person_with_tracking(self, img: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        """可视化多人姿态和tracking ID - 使用Numba加速的绘制功能

        Args:
            img: 原始图像
            tracked_objects: 包含tracking信息的对象列表

        Returns:
            可视化后的图像
        """
        # 使用Numba加速的绘制函数
        return draw_multi_person_pose_numba(img, tracked_objects, self.pose_score_thr)

    # real batch process
    @timeit(name=f'{__name__}.inference_batch_images ', log_threshold_ms=50)
    def inference_batch_images(self, imgs: List[np.ndarray]) -> List[
        Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]]:
        """
        pose estimator and tracking ID in  a  batch
        :param imgs:
        :return: List[Tuple[np.ndarray, np.ndarray]] (img,keypoints,score)
        """
        if not imgs:
            return [], [], []

        batch_detections = self._batch_detect_humans(imgs)  # 40 ms
        # 收集所有需要进行姿态估计的人体区域
        all_person_imgs = []
        all_centers = []
        all_scales = []
        image_person_counts = []  # 记录每张图像有多少人
        img_bbox_mapping = []  # 记录每个人体区域属于哪张图像
        with time_block('pose_estimator', log_result=True):  # 25
            for img_idx, (img, bboxes, track_ids) in enumerate(batch_detections):
                image_person_counts.append(len(bboxes))

                for bbox_idx, bbox in enumerate(bboxes):
                    person_img = self._crop_person(img, bbox)
                    resized_img, center, scale = self.pose_estimator.preprocess(person_img)
                    resized_img = resized_img.astype(np.float32)

                    all_person_imgs.append(resized_img)
                    all_centers.append(center)
                    all_scales.append(scale)
                    img_bbox_mapping.append((img_idx, bbox_idx, bbox))

        # 如果有人体区域，进行批量ONNX推理
        with time_block('run_inference_batch', log_result=True):  # 168 ms
            if all_person_imgs:
                # 构建超大batch输入
                mega_batch_input = np.stack(all_person_imgs, axis=0)

                # 批量ONNX推理
                mega_batch_outputs = self.pose_estimator.run_inference_batch(mega_batch_input)

                # 分解batch结果
                all_keypoints = []
                all_scores = []

                for i, (img_idx, bbox_idx, bbox) in enumerate(img_bbox_mapping):
                    single_outputs = [output[i:i + 1] for output in mega_batch_outputs]
                    keypoints, scores = self.pose_estimator.postprocess(
                        single_outputs,
                        self.pose_estimator.model_input_size,
                        all_centers[i],
                        all_scales[i]
                    )
                    keypoints = self._transform_keypoints_to_original(keypoints, bbox)
                    all_keypoints.append(keypoints)
                    all_scores.append(scores)

            # 重新组织结果按图像分组
            results = []
            person_idx = 0

        for img_idx, (img, bboxes, track_ids) in enumerate(batch_detections):
            if len(bboxes) == 0:
                # 没有检测到人的图像
                if self.enable_tracking and not self.use_yolo_tracking:
                    self.iou_tracker.track_persons([])
                results.append((img, [], []))
                continue

            # 获取当前图像的所有检测结果
            current_detections = []
            for bbox_idx, bbox in enumerate(bboxes):
                current_detections.append({
                    'bbox': bbox[:4],
                    'keypoints': all_keypoints[person_idx],
                    'scores': all_scores[person_idx]
                })
                person_idx += 1

            # Tracking处理
            if self.use_yolo_tracking and track_ids:
                tracked_objects = []
                for detection, track_id in zip(current_detections, track_ids):
                    tracked_objects.append({
                        'track_id': track_id,
                        'bbox': detection['bbox'],
                        'keypoints': detection['keypoints'],
                        'scores': detection['scores'],
                        'age': 0
                    })
            else:
                tracked_objects = self.iou_tracker.track_persons(current_detections)

            # 可视化

            vis_img = self._visualize_multi_person_with_tracking(img, tracked_objects)

            # 格式化返回结果
            keypoints_list = [obj['keypoints'] for obj in tracked_objects]
            scores_list = [obj['scores'] for obj in tracked_objects]

            results.append((vis_img, keypoints_list, scores_list))

        return results

    def _batch_detect_humans(self, imgs: List[np.ndarray]) -> List[
        Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]]:
        """批量检测人体"""
        results = []

        if self.detector_type == 'yolo':
            if batch_settings.use_bytetrack:
                # 使用ByteTrack跟踪
                if not hasattr(self, '_detector_tracker'):
                    self._detector_tracker = DetectorWithTracking()

                batch_tracks = self._detector_tracker.detect_and_track_batch(imgs)
                for img, tracks in zip(imgs, batch_tracks):
                    bboxes = []
                    track_ids = []
                    for track in tracks:
                        bbox = np.array(track['bbox'] + [track['score']])
                        bboxes.append(bbox)
                        track_ids.append(track['track_id'])
                    results.append((img, bboxes, track_ids))
            else:
                # 仅检测，不跟踪
                batch_bboxes = self._batch_detect_yolo(imgs)
                for img, bboxes in zip(imgs, batch_bboxes):
                    results.append((img, bboxes, None))
        else:
            # MMDetection批量检测
            if self._can_batch_detect_mmdet():
                batch_bboxes = self._batch_detect_mmdet(imgs)
                for img, bboxes in zip(imgs, batch_bboxes):
                    results.append((img, bboxes, None))
            else:
                # 逐个检测
                for img in imgs:
                    bboxes = self._detect_humans(img)
                    results.append((img, bboxes, None))

        return results

    def _batch_detect_yolo(self, imgs: List[np.ndarray]) -> List[List[np.ndarray]]:
        """批量YOLO检测"""
        if not hasattr(self, '_yolo_detector'):
            from app.pose_service.yolo_detector import YOLODetector
            self._yolo_detector = YOLODetector()

        # 批量检测
        batch_detections = self._yolo_detector.detect_batch(imgs)

        # 转换格式
        batch_bboxes = []
        for detections in batch_detections:
            bboxes = []
            for detection in detections:
                bbox = np.array(detection['bbox'] + [detection['score']])
                if bbox[4] > self.det_score_thr:  # 检查置信度阈值
                    bboxes.append(bbox)
            batch_bboxes.append(bboxes)

        return batch_bboxes

    def _can_batch_detect_mmdet(self) -> bool:
        """检查MMDetection是否支持batch检测"""
        # 大多数MMDetection模型支持batch推理
        return self.detector_type == 'mmdet'

    def _batch_detect_mmdet(self, imgs: List[np.ndarray]) -> List[List[np.ndarray]]:
        """MMDetection批量检测"""

        results = inference_detector(self.detector, imgs)

        batch_bboxes = []
        for result in results:
            bboxes = self._extract_bboxes_from_mmdet_result(result)
            batch_bboxes.append(bboxes)

        return batch_bboxes
