import asyncio
import logging
import os
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# from app.api.v1.api import api_router
from app.config import settings
from app.video_service import VideoTransformTrack
from app.video_service.video_process import (
    pcs, register_video_track, unregister_video_track,
    get_global_tracking_status, set_global_tracking_enabled, reset_global_tracking
)

# 启用日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

from fastapi.staticfiles import StaticFiles

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True)
app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI application for MMPose integration with tracking",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
# app.include_router(api_router, prefix=settings.API_V1_STR)
# 挂载 uploads 路由
app.mount(
    "/uploads",
    StaticFiles(directory=os.path.join("app", "uploads")),
    name="uploads"
)
app.mount('/static', StaticFiles(directory=str(BASE_DIR / "static")), name='static')


@app.post("/upload")
async def upload_file(request: Request):
    form = await request.form()
    file = form.get("file")
    if not file:
        return JSONResponse(status_code=400, content={"detail": "No file uploaded"})
    if not file.filename.endswith(('.mp4', '.webm', '.ogg', '.mov', '.avi')):
        return JSONResponse(status_code=400, content={"detail": "Unsupported file type"})
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info(f"File uploaded successfully: {file_location}")
    return {"filename": file.filename, "location": file_location}


@app.get("/uploads")
async def list_uploads():
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            files.append({"filename": f.name, "size": f.stat().st_size})
    return {"files": files}


@app.delete("/uploads/{filename}")
async def delete_upload(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"detail": "Deleted"}
    return JSONResponse(status_code=404, content={"detail": "File not found"})


@app.post("/offer")
async def offer(request: Request):
    logger.info(f"Received offer request")
    data = await request.json()
    offers = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    mode = data.get("mode", "camera")
    pc = RTCPeerConnection()
    pcs.add(pc)

    # 添加连接状态监听
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            # 清理video track
            if hasattr(pc, "_video_track") and pc._video_track:
                unregister_video_track(pc._video_track)
            # 停止并清理 MediaPlayer
            if hasattr(pc, "_player") and pc._player:
                logger.info("Stopping MediaPlayer tracks")
                if pc._player.audio:
                    pc._player.audio.stop()
                if pc._player.video:
                    pc._player.video.stop()
                pc._player = None
            await pc.close()
            pcs.discard(pc)

    # 添加ICE连接状态监听
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")

    # 先设置远程描述

    # Handle different modes
    if mode == "camera":
        @pc.on("track")
        def on_track(track):
            logger.info(f"Track {track.kind} received")
            if track.kind == "video":
                transformed = VideoTransformTrack(track)
                pc.addTrack(transformed)
                # 注册video track以便tracking控制
                register_video_track(transformed)
                pc._video_track = transformed  # 保存引用用于清理
                logger.info("Video track transformed and added with tracking support")

        await pc.setRemoteDescription(offers)
    elif mode == "upload":
        filename = data.get("fileName")
        if not filename:
            return JSONResponse(status_code=400, content={"detail": "No fileName provided"})

        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            return JSONResponse(status_code=400, content={"detail": "File not found"})
        try:
            player = MediaPlayer(str(file_path), loop=True)
            pc._player = player
            if player.video:
                transformed = VideoTransformTrack(player.video)
                pc.addTrack(transformed)
                # 注册video track以便tracking控制
                register_video_track(transformed)
                pc._video_track = transformed  # 保存引用用于清理
                logger.info("Video track added using MediaPlayer with tracking support")
                await pc.setRemoteDescription(offers)
            else:
                return JSONResponse(status_code=400, content={"detail": "No video track in file"})

        except Exception as e:
            logger.error(f"Failed to open video file: {e}")
            return JSONResponse(status_code=400, content={"detail": f"Cannot open video file: {str(e)}"})

    elif mode == "stream":
        stream_url = data.get("streamUrl")
        if not stream_url:
            return JSONResponse(status_code=400, content={"detail": "No streamUrl provided"})

        try:
            player = MediaPlayer(stream_url, format="ffmpeg", options={"rtsp_transport": "tcp"})
            pc._player = player  # 保存引用以便清理
            await pc.setRemoteDescription(offers)
            if player.video:
                transformed = VideoTransformTrack(player.video)
                pc.addTrack(transformed)
                # 注册video track以便tracking控制
                register_video_track(transformed)
                pc._video_track = transformed  # 保存引用用于清理
                logger.info("Video track from stream added with tracking support")
            else:
                return JSONResponse(status_code=400, content={"detail": "No video track in stream"})
        except Exception as e:
            logger.error(f"Failed to open stream: {e}")
            return JSONResponse(status_code=400, content={"detail": f"Cannot open stream: {str(e)}"})

    else:
        return JSONResponse(status_code=400, content={"detail": f"Unknown mode {mode}"})

    # 创建并发送 answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Answer created and local description set")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/")
async def index():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.on_event("shutdown")
async def on_shutdown():
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Tracking control API endpoints
@app.post("/tracking/reset")
async def reset_tracking():
    """重置所有活跃连接的tracking功能"""
    reset_global_tracking()
    return {"status": "success", "message": "All tracking reset"}


@app.post("/tracking/toggle")
async def toggle_tracking(enabled: bool = True):
    """启用或禁用所有连接的tracking功能"""
    set_global_tracking_enabled(enabled)
    return {"status": "success", "enabled": enabled, "message": f"Tracking {'enabled' if enabled else 'disabled'} for all connections"}


@app.get("/tracking/status")
async def get_tracking_status():
    """获取全局tracking状态"""
    status = get_global_tracking_status()
    return {
        "enabled": status["enabled"],
        "active_video_tracks": status["active_tracks"],
        "active_connections": len(pcs)
    }
