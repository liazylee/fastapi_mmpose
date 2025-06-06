import asyncio
import logging
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.kafaka_producer import kafka_frame_producer
from app.routers import webrtc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Create directories
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI application for MMPose integration with streaming pose estimation",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

# Include WebRTC router with all streaming functionality
app.include_router(webrtc.router, prefix="/webrtc", tags=["WebRTC"])


# File management endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), bg_tasks: BackgroundTasks = None):
    """Upload video file for processing"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    bg_tasks.add_task(kafka_frame_producer, file_path)
    return {"filename": file.filename, "detail": "File uploaded and processing started"}


@app.get("/uploads")
async def list_uploads():
    """List all uploaded files"""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })
    return {"files": files}


@app.delete("/uploads/{filename}")
async def delete_upload(filename: str):
    """Delete uploaded file"""
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        logger.info(f"File deleted: {filename}")
        return {"detail": "File deleted successfully"}
    return JSONResponse(status_code=404, content={"detail": "File not found"})


# Main page
@app.get("/")
async def index():
    """Return the main application page"""
    return templates.TemplateResponse("index.html", {"request": {}})


# Application lifecycle events
@app.on_event("startup")
async def on_startup():
    """Application startup tasks"""
    logger.info("🚀 FastAPI MMPose Streaming Application starting up...")

    logger.info("WebRTC streaming processor initialized")


@app.on_event("shutdown")
async def on_shutdown():
    """Application shutdown tasks"""
    logger.info("🛑 FastAPI MMPose Streaming Application shutting down...")

    # Clean up WebRTC connections
    from app.video_service.video_process import pcs
    if pcs:
        logger.info(f"Closing {len(pcs)} WebRTC connections...")
        close_tasks = [pc.close() for pc in pcs]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        pcs.clear()
        logger.info("All WebRTC connections closed")

    logger.info("Application shutdown complete")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    from app.routers.webrtc import active_connections, active_video_tracks

    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "version": "2.0.0",
        "active_connections": len(active_connections),
        "active_video_tracks": len(active_video_tracks),
        "upload_directory": str(UPLOAD_DIR),
        "features": [
            "WebRTC streaming",
            "Real-time pose estimation",
            "Batch processing",
            "Tracking support",
            "Performance monitoring"
        ]
    }


# API documentation endpoints


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
