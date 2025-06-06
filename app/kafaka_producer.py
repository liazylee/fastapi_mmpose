import asyncio
import time

import cv2
import msgpack
from aiokafka import AIOKafkaProducer

from app.config import batch_settings

KAFKA_BOOTSTRAP_SERVERS = 'kafka:9092'
KAFKA_TOPIC = 'video_raw_frames'


async def kafka_frame_producer(video_path):
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await producer.start()

    cap = cv2.VideoCapture(video_path)
    batch_size = batch_settings.batch_size  # define batch_settings elsewhere
    frame_batch = []

    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            _, buffer = cv2.imencode('.jpg', frame)
            frame_batch.append({
                "timestamp": int(time.time() * 1000),
                "frame": buffer.tobytes()
            })

            if len(frame_batch) >= batch_size:
                # Serialize the batch and send
                payload = msgpack.packb(frame_batch)
                await producer.send_and_wait(KAFKA_TOPIC, payload)
                frame_batch.clear()

            # Optional: yield to event loop
            await asyncio.sleep(0)

        # Send the final incomplete batch if any
        if frame_batch:
            payload = msgpack.packb(frame_batch)
            await producer.send_and_wait(KAFKA_TOPIC, payload)

    finally:
        await producer.stop()
        cap.release()
