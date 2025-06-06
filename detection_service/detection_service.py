from typing import List, Dict

import cv2
import msgpack
import numpy as np
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from .yolo_detector import DetectorWithTracking

KAFKA_BOOTSTRAP_SERVERS = 'kafka:9092'
KAFKA_TOPIC = 'video_raw_frames'
yolo_tracker = DetectorWithTracking()


async def yolo_detection_service():
    consumer = AIOKafkaConsumer(KAFKA_TOPIC, KAFKA_BOOTSTRAP_SERVERS, group_id='detection_service')
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await consumer.start()
    await producer.start()
    try:
        async for msg in consumer:
            frame_batch = msgpack.unpackb(msg.value, raw=False)  # 解码为 List[Dict]
            frames = []
            timestamps = []
            for item in frame_batch:
                frame_bytes = item['frame']
                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                frames.append(frame)
                timestamps.append(item['timestamp'])
            tracks_batch = detect_and_track_batch(frames)
            processed_batch = []

            for i in range(len(frames)):
                processed_batch.append({
                    "timestamp": timestamps[i],
                    "frame": frame_batch[i]["frame"],  # 原始JPEG字节
                    "metadata": {
                        "tracks": tracks_batch[i]  # list of {id, bbox, conf}
                    }
                })
            await producer.send_and_wait("video_detected_frames", msgpack.packb(processed_batch))

    finally:
        await consumer.stop()
        await producer.stop()


def detect_and_track_batch(frames: List[np.ndarray]) -> List[List[Dict]]:
    """

    :param frames:
    :return: list of {id, bbox, conf}
    """
    batch_tracks = yolo_tracker.detect_and_track_batch(frames)

    # 标准化输出格式为 List[Dict]
    standardized_batch_tracks = []

    for frame_tracks in batch_tracks:
        standardized_tracks = []
        for track in frame_tracks:
            standardized_tracks.append({
                "id": track.get('track_id', -1),  # 默认-1表示未分配track_id
                "bbox": track['bbox'],
                "conf": track['score']
            })
        standardized_batch_tracks.append(standardized_tracks)

    return standardized_batch_tracks
