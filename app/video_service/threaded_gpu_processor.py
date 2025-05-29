import logging
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector
from mmpose.apis import inference_topdown

from app.config import batch_settings, settings

logger = logging.getLogger(__name__)
try:
    from app.video_service.gpu_stabilizer import stabilizer
except ImportError:
    print("GPU stabilizer not found, running without stabilization")
    stabilizer = None


class StreamedGPUProcessor:
    """GPU processor using CUDA streams for maximum performance"""

    def __init__(self, config=batch_settings):
        self.config = config
        self.device = torch.device('cuda:0')
        # Initialize buffers FIRST
        self.detection_buffer = deque(maxlen=3)
        self.pose_buffer = deque(maxlen=3)

        # Statistics
        self.processed_frames = 0
        self.start_time = time.time()

        # Create CUDA streams
        self.detection_stream = torch.cuda.Stream()
        self.pose_stream = torch.cuda.Stream()
        self.viz_stream = torch.cuda.Stream()

        # Initialize models LAST
        self._init_models()

        if stabilizer:
            stabilizer.warmup_streamed_processor(self, num_warmup=15)

    def _init_models(self):
        """Initialize models on GPU"""
        from app.pose_service.core import pose_service
        self.detector = pose_service.detector
        self.pose_estimator = pose_service.pose_estimator
        self.visualizer = pose_service.visualizer

        # Warm up GPU
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.process_single_frame(dummy_frame)

    def process_single_frame(self, frame):
        """Process single frame using GPU streams pipeline"""
        result = None

        # Stage 1: Detection (Stream 1)
        with torch.cuda.stream(self.detection_stream):
            detection_result = inference_detector(self.detector, frame)
            filtered_boxes = self._filter_detections(detection_result)
            self.detection_buffer.append((frame, filtered_boxes))

        # Stage 2: Pose (Stream 2) - process previous frame
        if len(self.detection_buffer) >= 2:
            with torch.cuda.stream(self.pose_stream):
                prev_frame, prev_detection = self.detection_buffer[0]
                if len(prev_detection) > 0:
                    pose_result = inference_topdown(
                        self.pose_estimator, prev_frame, prev_detection
                    )
                else:
                    pose_result = []
                self.pose_buffer.append((prev_frame, pose_result))

        # Stage 3: Visualization (Stream 3) - process older frame
        if len(self.pose_buffer) >= 1:
            with torch.cuda.stream(self.viz_stream):
                old_frame, old_pose = self.pose_buffer.popleft()
                # data_samples = merge_data_samples(old_pose)
                # self.visualizer.add_datasample(
                #     'result',
                #     old_frame,
                #     data_sample=old_pose,
                #     draw_gt=False,
                #     draw_heatmap=False,  # Do not draw heatmap
                #     draw_bbox=True,
                #     show_kpt_idx=True,
                #     skeleton_style='mmpose',
                #     show=False,
                #     wait_time=0,
                #     kpt_thr=0.3
                # )
                if old_pose:
                    result = self._draw_poses_manually(old_frame, old_pose)
                else:
                    result = old_frame

                # Update stats
                self.processed_frames += 1
                if stabilizer:
                    stabilizer.on_frame_processed()
                # Log performance every 100 frames
                if self.processed_frames % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.processed_frames / elapsed
                    logger.info(f"GPU Streams FPS: {fps:.1f}")

        return result if result is not None else frame

    def _filter_detections(self, det_result):
        """Filter detection results"""
        try:
            pred_instance = det_result.pred_instances.cpu().numpy()
            if pred_instance is None or len(pred_instance) == 0:
                return np.array([]).reshape(0, 4)

            person_mask = np.logical_and(
                pred_instance.labels == settings.det_cat_id,
                pred_instance.scores > settings.bbox_thr
            )

            if not person_mask.any():
                return np.array([]).reshape(0, 4)

            boxes = pred_instance.bboxes[person_mask]
            return boxes

        except Exception:
            return np.array([]).reshape(0, 4)

    def _draw_poses_manually(self, image, pose_results):
        """Manual pose drawing"""
        vis_img = image.copy()

        keypoint_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
        ]

        default_skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
        ]

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

                        # Draw keypoints
                        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                            if score > 0.3:
                                x, y = int(kpt[0]), int(kpt[1])
                                if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                                    color = keypoint_colors[i % len(keypoint_colors)]
                                    cv2.circle(vis_img, (x, y), 2, color, -1)
                                    cv2.circle(vis_img, (x, y), 3, (255, 255, 255), 1)

                        # Draw skeleton
                        for connection in default_skeleton:
                            pt1_idx, pt2_idx = connection
                            if (pt1_idx < len(scores) and pt2_idx < len(scores) and
                                    scores[pt1_idx] > 0.3 and scores[pt2_idx] > 0.3):
                                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                                if (0 <= pt1[0] < vis_img.shape[1] and 0 <= pt1[1] < vis_img.shape[0] and
                                        0 <= pt2[0] < vis_img.shape[1] and 0 <= pt2[1] < vis_img.shape[0]):
                                    cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)

        return vis_img


class ThreadedPoseService:
    """Threaded pose service without async overhead"""

    def __init__(self, config=batch_settings):
        self.config = config

        # Simple queues (not asyncio)
        self.input_queue = queue.Queue(maxsize=config.max_queue_size)
        self.output_queue = queue.Queue(maxsize=config.max_queue_size)

        # GPU processor
        self.gpu_processor = StreamedGPUProcessor(config)

        # Worker thread
        self.running = False
        self.worker_thread = None

        # Batch collection
        self.batch_frames = []
        self.batch_size = min(config.batch_size, 16)  # Limit batch size for low latency

    def start(self):
        """Start the processing thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()

    def process_frame(self, frame):
        """Synchronous frame processing"""
        if not self.running:
            self.start()

        try:
            # Put frame in queue
            self.input_queue.put(frame, timeout=0.1)

            # Get result
            result = self.output_queue.get(timeout=0.5)
            return result, None

        except queue.Empty:
            # Return original frame if processing is slow
            return frame, None

    def _worker_loop(self):
        """Main worker loop"""
        while self.running:
            try:
                # Collect batch
                frames = []

                # Get first frame (blocking)
                frame = self.input_queue.get(timeout=1.0)
                frames.append(frame)

                # Get additional frames (non-blocking)
                for _ in range(self.batch_size - 1):
                    try:
                        frame = self.input_queue.get_nowait()
                        frames.append(frame)
                    except queue.Empty:
                        break

                # Process batch
                for frame in frames:
                    result = self.gpu_processor.process_single_frame(frame)

                    # Put result in output queue
                    try:
                        self.output_queue.put(result, timeout=0.1)
                    except queue.Full:
                        # Skip if output queue is full
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker loop error: {e}")
