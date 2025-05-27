from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse.

router = APIRouter()


@router.post("/upload")
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


@router.get("/uploads")
async def list_uploads():
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            files.append({"filename": f.name, "size": f.stat().st_size})
    return {"files": files}


@router.delete("/uploads/{filename}")
async def delete_upload(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"detail": "Deleted"}
    return JSONResponse(status_code=404, content={"detail": "File not found"})


@router.post("/offer")
async def offer(request: Request):
    logger.info(f"Received offer request")
    data = await request.json()
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    mode = data.get("mode", "camera")
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
    if mode == "camera":
        @pc.on("track")
        def on_track(track):
            logger.info(f"Track {track.kind} received")
            if track.kind == "video":
                transformed = VideoTransformTrack(track)
                pc.addTrack(transformed)
                logger.info("Video track transformed and added")
            elif track.kind == "audio":
                pc.addTrack(track)
                logger.info("Audio track added")
    elif mode == "upload":
        filename = data.get("fileName")
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            return JSONResponse(status_code=400, content={"detail": "File not found"})
        player = MediaPlayer(str(file_path))
        pc.addTransceiver("video", direction="sendonly")
        if player.video:
            pc.addTrack(VideoTransformTrack(player.video))
        if player.audio:
            pc.addTrack(player.audio)

    elif mode == "stream":
        stream_url = data.get("streamUrl")
        if not stream_url:
            return JSONResponse(status_code=400, content={"detail": "No streamUrl provided"})
        player = MediaPlayer(stream_url, format="ffmpeg", options={"rtsp_transport": "tcp"})
        pc.addTransceiver("video", direction="sendonly")
        if player.video:
            pc.addTrack(VideoTransformTrack(player.video))
        if player.audio:
            pc.addTrack(player.audio)

    else:
        return JSONResponse(status_code=400, content={"detail": f"Unknown mode {mode}"})

    # Set remote offer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Answer created and local description set")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
