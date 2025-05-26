import asyncio
import logging
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.api.v1.api import api_router
from app.config import settings
from app.video_service.video_process import pcs, VideoTransformTrack

# 启用日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI application for MMPose integration",
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
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.post("/upload")
async def upload_file(request: Request):
    form = await request.form()
    file = form.get("file")
    if not file:
        return JSONResponse(status_code=400, content={"detail": "No file uploaded"})
    if not file.filename.endswith(('.mp4', '.webm', '.ogg', '.mov', '.avi')):
        return JSONResponse(status_code=400, content={"detail": "Unsupported file type"})
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info(f"File uploaded successfully: {file_location}")
    return {"filename": file.filename, "location": file_location}


@app.post("/offer")
async def offer(request: Request):
    logger.info(f"Received offer request: {request}")
    data = await request.json()
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # 添加连接状态监听
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # 添加ICE连接状态监听
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")

    # Handle incoming tracks
    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        if track.kind == "video":
            # Wrap incoming track with our transform
            transformed = VideoTransformTrack(track)
            pc.addTrack(transformed)
            logger.info("Video track transformed and added")
        # audio can be blackholed
        elif track.kind == "audio":
            pc.addTrack(track)
            logger.info("Audio track added")

    # Set remote offer
    await pc.setRemoteDescription(offer)
    logger.info("Remote description set")

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Answer created and local description set")

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


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
