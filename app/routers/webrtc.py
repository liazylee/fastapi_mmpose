import logging
import uuid
from typing import Dict

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.video_service.video_process import (
    pcs,
    AsyncVideoTransformTrack as VideoTransformTrack,
    register_video_track,
    unregister_video_track,
    get_global_tracking_status,
    set_global_tracking_enabled,
    reset_global_tracking,
    cleanup_pcs
)

# Create the API router
router = APIRouter()
relay = MediaRelay()
KAFKA_BOOTSTRAP_SERVERS = 'kafka:9092'
KAFKA_TOPIC = 'video_raw_frames'
# Track active connections and video tracks
active_connections: Dict[str, RTCPeerConnection] = {}
active_video_tracks: Dict[str, VideoTransformTrack] = {}

logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
async def get_index():
    """Return the HTML page"""
    with open("static/webrtc.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)


@router.post("/offer")
async def offer(request: Request):
    """Handle WebRTC offer and create answer - supports camera, upload, and stream modes"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    mode = params.get("mode", "camera")

    connection_id = str(uuid.uuid4())
    pc = RTCPeerConnection()
    pcs.add(pc)
    active_connections[connection_id] = pc

    logger.info(f"Received {mode} mode offer request for connection {connection_id[:8]}")

    # Set up connection handlers
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection {connection_id[:8]} state: {pc.connectionState}")

        if pc.connectionState in ("failed", "closed"):
            # Clean up when connection closes
            if connection_id in active_connections:
                del active_connections[connection_id]

            # Stop and clean up MediaPlayer (for upload and stream modes)
            if hasattr(pc, "_player") and pc._player:
                logger.info("Stopping MediaPlayer tracks")
                if pc._player.audio:
                    pc._player.audio.stop()
                if pc._player.video:
                    pc._player.video.stop()
                pc._player = None

            # Clean up video track
            if connection_id in active_video_tracks:
                video_track = active_video_tracks[connection_id]
                # Properly await the async stop method
                try:
                    await video_track.stop()
                except Exception as e:
                    logger.warning(f"Error stopping video track: {e}")

                unregister_video_track(video_track)
                del active_video_tracks[connection_id]

            # Clean up from global pcs set
            if pc in pcs:
                pcs.remove(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")

    # Handle different modes
    if mode == "camera":
        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            logger.info(f"Track {track.kind} received")

            if track.kind == "video":
                # Create async video transform track with 30 FPS target
                video_track = VideoTransformTrack(relay.subscribe(track), target_fps=30)
                active_video_tracks[connection_id] = video_track
                register_video_track(video_track)

                # Add the transformed track to the connection
                pc.addTrack(video_track)

                logger.info(f"Video track created for camera connection {connection_id[:8]}")

        # @track.on("ended")
        # async def on_ended():
        #     logger.info(f"Track {track.kind} ended for connection {connection_id[:8]}")

        await pc.setRemoteDescription(offer)

    elif mode == "upload":
        filename = params.get("fileName")
        loop_playback = params.get("loop", False)  # Default to no loop

        if not filename:
            return JSONResponse(status_code=400, content={"detail": "No fileName provided"})

        # Import here to get the upload directory from main.py
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent.parent
        UPLOAD_DIR = BASE_DIR / "uploads"

        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            return JSONResponse(status_code=400, content={"detail": "File not found"})

        try:
            from aiortc.contrib.media import MediaPlayer
            player = MediaPlayer(str(file_path), loop=loop_playback)
            pc._player = player  # Save reference for cleanup

            if player.video:
                # Create async video transform track
                video_track = VideoTransformTrack(player.video, target_fps=30)
                active_video_tracks[connection_id] = video_track
                register_video_track(video_track)

                # Add track ended handler
                @player.video.on("ended")
                async def on_video_ended():
                    logger.info(f"Video file {filename} playback completed for connection {connection_id[:8]}")
                    # Mark track as ended and let the connection close naturally
                    if connection_id in active_video_tracks:
                        video_track = active_video_tracks[connection_id]
                        video_track._track_ended = True

                pc.addTrack(video_track)
                logger.info(f"Video track added using MediaPlayer for file: {filename} (loop: {loop_playback})")

                await pc.setRemoteDescription(offer)
            else:
                return JSONResponse(status_code=400, content={"detail": "No video track in file"})

        except Exception as e:
            logger.error(f"Failed to open video file {filename}: {e}")
            return JSONResponse(status_code=400, content={"detail": f"Cannot open video file: {str(e)}"})

    elif mode == "stream":
        stream_url = params.get("streamUrl")
        if not stream_url:
            return JSONResponse(status_code=400, content={"detail": "No streamUrl provided"})

        try:
            from aiortc.contrib.media import MediaPlayer
            player = MediaPlayer(stream_url, format="ffmpeg", options={"rtsp_transport": "tcp"})
            pc._player = player  # Save reference for cleanup

            await pc.setRemoteDescription(offer)

            if player.video:
                # Create async video transform track
                video_track = VideoTransformTrack(player.video, target_fps=30)
                active_video_tracks[connection_id] = video_track
                register_video_track(video_track)

                # Add track ended handler for streams
                @player.video.on("ended")
                async def on_stream_ended():
                    logger.info(f"Stream {stream_url} ended for connection {connection_id[:8]}")
                    # Mark track as ended and let the connection close naturally
                    if connection_id in active_video_tracks:
                        video_track = active_video_tracks[connection_id]
                        video_track._track_ended = True

                pc.addTrack(video_track)
                logger.info(f"Video track from stream added: {stream_url}")
            else:
                return JSONResponse(status_code=400, content={"detail": "No video track in stream"})

        except Exception as e:
            logger.error(f"Failed to open stream {stream_url}: {e}")
            return JSONResponse(status_code=400, content={"detail": f"Cannot open stream: {str(e)}"})

    else:
        return JSONResponse(status_code=400, content={"detail": f"Unknown mode: {mode}"})

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    response_data = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "connection_id": connection_id
    }

    logger.info(f"WebRTC connection established ({mode} mode): {connection_id[:8]}")
    return response_data


# Tracking control endpoints
@router.get("/tracking/status")
async def get_tracking_status():
    """Get current tracking status"""
    return get_global_tracking_status()


@router.post("/tracking/enable")
async def enable_tracking():
    """Enable tracking for all connections"""
    set_global_tracking_enabled(True)
    return {"status": "Tracking enabled globally"}


@router.post("/tracking/disable")
async def disable_tracking():
    """Disable tracking for all connections"""
    set_global_tracking_enabled(False)
    return {"status": "Tracking disabled globally"}


@router.post("/tracking/reset")
async def reset_tracking():
    """Reset tracking for all connections"""
    reset_global_tracking()
    return {"status": "Tracking reset globally"}


# Performance monitoring endpoints
@router.get("/stats/connections")
async def get_connection_stats():
    """Get statistics for all active connections"""
    stats = {
        "total_connections": len(active_connections),
        "active_video_tracks": len(active_video_tracks),
        "connection_states": {},
        "processing_stats": {}
    }

    # Get connection states
    for conn_id, pc in active_connections.items():
        stats["connection_states"][conn_id[:8]] = pc.connectionState

    # Get video track processing stats
    for conn_id, video_track in active_video_tracks.items():
        if hasattr(video_track, 'get_performance_stats'):
            track_stats = video_track.get_performance_stats()
            stats["processing_stats"][conn_id[:8]] = track_stats

    return stats


@router.get("/stats/system")
async def get_system_stats():
    """Get comprehensive system statistics"""
    system_stats = {
        "connections": {
            "total": len(active_connections),
            "active_tracks": len(active_video_tracks)
        },
        "tracking": get_global_tracking_status(),
        "video_tracks": {}
    }

    # Get detailed stats from each video track
    for conn_id, video_track in active_video_tracks.items():
        if hasattr(video_track, 'get_performance_stats'):
            track_stats = video_track.get_performance_stats()
            system_stats["video_tracks"][conn_id[:8]] = track_stats

    return system_stats


@router.post("/cleanup")
async def cleanup_connections():
    """Clean up closed connections"""
    cleaned_pcs = cleanup_pcs()

    # Clean up active_connections dict
    closed_connections = []
    for conn_id, pc in list(active_connections.items()):
        if pc.connectionState == 'closed':
            closed_connections.append(conn_id)
            del active_connections[conn_id]

    # Clean up active_video_tracks dict
    closed_tracks = []
    for conn_id in list(active_video_tracks.keys()):
        if conn_id in closed_connections:
            video_track = active_video_tracks[conn_id]
            try:
                await video_track.stop()
            except Exception as e:
                logger.warning(f"Error stopping video track {conn_id[:8]}: {e}")

            unregister_video_track(video_track)
            del active_video_tracks[conn_id]
            closed_tracks.append(conn_id)

    return {
        "cleaned_pcs": cleaned_pcs,
        "closed_connections": len(closed_connections),
        "closed_tracks": len(closed_tracks),
        "remaining_connections": len(active_connections),
        "remaining_tracks": len(active_video_tracks)
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": len(active_connections),
        "active_video_tracks": len(active_video_tracks),
        "global_tracking_enabled": get_global_tracking_status()["enabled"]
    }


# Media source endpoints
@router.get("/media/uploads")
async def get_available_uploads():
    """Get list of available uploaded files for streaming"""
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"

    if not UPLOAD_DIR.exists():
        return {"files": []}

    files = []
    video_extensions = {'.mp4', '.webm', '.ogg', '.mov', '.avi', '.mkv', '.flv'}

    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
                "extension": f.suffix.lower(),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            })

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['modified'], reverse=True)

    return {
        "files": files,
        "total_files": len(files),
        "upload_directory": str(UPLOAD_DIR)
    }


@router.get("/media/stream/test")
async def test_stream_url():
    """Get test stream URLs for testing stream mode"""
    return {
        "test_streams": [
            {
                "name": "Big Buck Bunny (MP4)",
                "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
                "type": "mp4"
            },
            {
                "name": "Test Pattern Stream",
                "url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
                "type": "mp4"
            }
        ],
        "note": "These are sample URLs for testing. Replace with your actual stream URLs."
    }
