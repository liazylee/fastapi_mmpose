// static/app.js
let pc;
let localStream;
let currentFile = null;

const local = document.getElementById('local');
const remote = document.getElementById('remote');
const status = document.getElementById('status');

function updateStatus(message) {
    status.textContent = message;
    console.log(message);
}

function updateSystemInfo(mode, source = '') {
    document.getElementById('currentMode').textContent = mode;
    document.getElementById('currentSource').textContent = source;
}

function onInputModeChange() {
    const mode = document.getElementById('inputMode').value;
    document.getElementById('streamControls').classList.add('hidden');
    document.getElementById('uploadControls').classList.add('hidden');
    if (mode === 'stream') {
        document.getElementById('streamControls').classList.remove('hidden');
    } else if (mode === 'upload') {
        document.getElementById('uploadControls').classList.remove('hidden');
        document.getElementById('fileListContainer').classList.remove('hidden');
        refreshFileList();
    }
    updateSystemInfo(mode);
}

function onFileSelected() {
    const fileInput = document.getElementById('videoFile');
    const uploadBtn = document.getElementById('uploadBtn');
    if (fileInput.files.length > 0) {
        uploadBtn.disabled = false;
        currentFile = fileInput.files[0];
        updateStatus(`File selected: ${currentFile.name} (${(currentFile.size / 1024 / 1024).toFixed(2)} MB)`);
    } else {
        uploadBtn.disabled = true;
        currentFile = null;
    }
}

async function uploadFile() {
    if (!currentFile) {
        updateStatus('No file selected');
        return;
    }
    const formData = new FormData();
    formData.append('file', currentFile);
    const progressContainer = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    progressContainer.classList.remove('hidden');
    progressBar.style.width = '0%';
    try {
        updateStatus('Uploading file...');
        const response = await fetch('/upload', {method: 'POST', body: formData});
        progressBar.style.width = '100%';
        if (response.ok) {
            const result = await response.json();
            updateStatus(`File uploaded successfully: ${result.filename}`);
            refreshFileList();
            setTimeout(() => {
                progressContainer.classList.add('hidden');
            }, 2000);
        } else {
            const error = await response.json();
            updateStatus(`Upload failed: ${error.detail}`);
        }
    } catch (error) {
        updateStatus(`Upload error: ${error.message}`);
    }
}

async function refreshFileList() {
    try {
        const response = await fetch('/uploads');
        if (response.ok) {
            const data = await response.json();
            displayFileList(data.files);
        } else {
            updateStatus('Failed to fetch file list');
        }
    } catch (error) {
        updateStatus(`Error fetching files: ${error.message}`);
    }
}

function displayFileList(files) {
    const fileList = document.getElementById('fileList');
    const allFilesList = document.getElementById('allFilesList');
    if (files.length === 0) {
        const emptyMsg = '<div style="text-align: center; color: #666; padding: 10px;">No files uploaded</div>';
        fileList.innerHTML = emptyMsg;
        allFilesList.innerHTML = emptyMsg;
        return;
    }
    const fileItems = files.map(file => `
        <div class="file-item">
            <div>
                <strong>${file.filename}</strong><br>
                <small>${(file.size / 1024 / 1024).toFixed(2)} MB</small>
            </div>
            <div>
                <button onclick="selectFile('${file.filename}')" class="btn-primary" style="margin-right: 5px;">Select</button>
            </div>
        </div>
    `).join('');
    fileList.innerHTML = fileItems;
    allFilesList.innerHTML = fileItems;
}

function selectFile(filename) {
    currentFile = {name: filename};
    updateStatus(`Selected file: ${filename}`);
    updateSystemInfo('upload', filename);
}

async function deleteFile(filename) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
    try {
        const response = await fetch(`/uploads/${filename}`, {method: 'DELETE'});
        if (response.ok) {
            updateStatus(`File deleted: ${filename}`);
            refreshFileList();
        } else {
            const error = await response.json();
            updateStatus(`Delete failed: ${error.error}`);
        }
    } catch (error) {
        updateStatus(`Delete error: ${error.message}`);
    }
}

async function startConnection() {
    try {
        updateStatus('Initializing connection...');
        document.getElementById('connectionStatus').textContent = 'Connecting...';
        const mode = document.getElementById('inputMode').value;
        if (pc) {
            pc.close();
            pc = null;
        }
        pc = new RTCPeerConnection({
            iceServers: [
                {urls: 'stun:stun.l.google.com:19302'},
                {urls: 'stun:stun1.l.google.com:19302'}
            ]
        });
        pc.ontrack = (event) => {
            if (event.streams[0]) {
                remote.srcObject = event.streams[0];
                updateStatus('Receiving processed video stream');
                document.getElementById('connectionStatus').textContent = 'Connected';
            }
        };
        pc.onconnectionstatechange = () => {
            document.getElementById('connectionStatus').textContent = pc.connectionState;
            if (pc.connectionState === 'failed') updateStatus('Connection failed');
        };
        const payload = {sdp: '', type: 'offer', mode: mode};
        if (mode === 'camera') {
            localStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
            local.srcObject = localStream;
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
            pc.addTransceiver('video', {direction: 'recvonly'});
            const offer = await pc.createOffer({offerToReceiveVideo: true, offerToReceiveAudio: false});
            await pc.setLocalDescription(offer);
            payload.sdp = pc.localDescription.sdp;
            payload.type = pc.localDescription.type;
            updateSystemInfo('camera', 'Local Camera');
        } else if (mode === 'upload') {
            if (!currentFile) {
                updateStatus('Please select a file first');
                return;
            }
            payload.fileName = currentFile.name;
            local.src = `/uploads/${currentFile.name}`;
            local.load();
            updateSystemInfo('upload', currentFile.name);
            const offer = await pc.createOffer({offerToReceiveVideo: true, offerToReceiveAudio: false});
            await pc.setLocalDescription(offer);
            payload.sdp = pc.localDescription.sdp;
            payload.type = pc.localDescription.type;
        } else if (mode === 'stream') {
            const streamUrl = document.getElementById('streamUrl').value.trim();
            if (!streamUrl) {
                updateStatus('Please enter a stream URL');
                return;
            }
            payload.streamUrl = streamUrl;
            local.src = streamUrl;
            local.load().catch(_ => {
            });
            updateSystemInfo('stream', streamUrl);
            const offer = await pc.createOffer({offerToReceiveVideo: true, offerToReceiveAudio: false});
            await pc.setLocalDescription(offer);
            payload.sdp = pc.localDescription.sdp;
            payload.type = pc.localDescription.type;
        }
        updateStatus('Sending connection request to server...');
        const response = await fetch('/webrtc/offer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const answer = await response.json();
        if (answer.error) {
            updateStatus(`Server error: ${answer.error}`);
            document.getElementById('connectionStatus').textContent = 'Error';
            return;
        }
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
        updateStatus(`${mode} processing started successfully`);
        document.getElementById('connectionStatus').textContent = 'Processing';
    } catch (err) {
        updateStatus(`Error: ${err.message}`);
        document.getElementById('connectionStatus').textContent = 'Error';
        console.error(err);
    }
}

function stopConnection() {
    if (pc) {
        pc.close();
        pc = null;
    }
    if (localStream) {
        localStream.getTracks().forEach(t => t.stop());
        localStream = null;
    }
    local.srcObject = null;
    local.src = '';
    remote.srcObject = null;
    updateStatus('Connection stopped');
    document.getElementById('connectionStatus').textContent = 'Disconnected';
    updateSystemInfo('None', 'None');
}

async function checkHealth() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            updateStatus(`System health: ${data.status}`);
        } else {
            updateStatus('Health check failed');
        }
    } catch (error) {
        updateStatus(`Health check error: ${error.message}`);
    }
}

document.addEventListener('DOMContentLoaded', function () {
    onInputModeChange();
    refreshFileList();
});