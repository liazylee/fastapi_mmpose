# FastAPI MMPose

A real-time human pose estimation web application built with FastAPI and MMPose, supporting multiple video input sources
including camera feeds, uploaded videos, and streaming URLs.

## 🌟 Features

- **Real-time Pose Estimation**: Powered by MMPose with RTMPose models for accurate human pose detection
- **Multiple Input Sources**:
    - Live camera feed
    - Video file uploads (MP4, WebM, AVI, MOV, MKV)
    - RTMP/RTSP/HTTP streaming URLs
- **WebRTC Integration**: Low-latency video streaming using aiortc
- **Web Dashboard**: Modern, responsive web interface for easy interaction
- **GPU Acceleration**: CUDA support for improved performance
- **Batch Processing**: Optimized for handling multiple frames efficiently
- **ONNX Support**: Includes ONNX runtime for optimized inference

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Computer Vision**: MMPose, MMDetection, OpenCV
- **Streaming**: WebRTC (aiortc), FFmpeg
- **Deep Learning**: PyTorch, ONNX Runtime
- **Frontend**: HTML5, JavaScript, CSS3
- **GPU**: CUDA support with TensorRT integration

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- FFmpeg installed on system
- Webcam (for camera mode)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fastapi_mmpose.git
cd fastapi_mmpose
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install MMPose (Development Version)

The project uses a specific version of MMPose:

```bash
pip install -e git+https://github.com/open-mmlab/mmpose.git@71ec36ebd63c475ab589afc817868e749a61491f#egg=mmpose
```

## 🏃‍♂️ Quick Start

### 1. Run the Application

```bash
python run.py
```

### 2. Access the Web Interface

Open your browser and navigate to: `http://localhost:8000`

### 3. Choose Your Input Source

- **Camera**: Use your local webcam
- **Upload**: Upload a video file
- **Stream**: Enter an RTMP/RTSP/HTTP stream URL

### 4. Start Processing

Click "Start Processing" to begin real-time pose estimation!

## 📁 Project Structure

```
fastapi_mmpose/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Configuration settings
│   ├── pose_service/          # Pose estimation core
│   │   ├── core.py           # Main pose processing logic
│   │   ├── onnx_inference.py # ONNX runtime inference
│   │   └── configs/          # Model configuration files
│   ├── video_service/         # Video processing pipeline
│   │   ├── video_process.py  # WebRTC video handling
│   │   └── batch_GPU_process_onnx.py # Batch processing
│   ├── templates/             # HTML templates
│   ├── static/               # CSS/JS assets
│   └── uploads/              # Uploaded video files
├── requirements.txt           # Python dependencies
├── run.py                    # Application launcher
├── topdown_demo_with_mmdet.py # Standalone demo script
└── README.md                 # This file
```

## ⚙️ Configuration

### Model Settings

The application uses RTMPose models by default:

- **Detector**: RTMDet-M for person detection
- **Pose Estimator**: RTMPose-M for pose estimation

### Device Configuration

- Automatically detects CUDA availability
- Falls back to CPU if GPU not available
- Configurable batch sizes and processing parameters

### Key Configuration Options (config.py)

```python
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
FPS = 30
BBOX_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
```

## 🎯 API Endpoints

| Endpoint              | Method | Description           |
|-----------------------|--------|-----------------------|
| `/`                   | GET    | Main web interface    |
| `/offer`              | POST   | WebRTC offer handling |
| `/upload`             | POST   | Video file upload     |
| `/uploads`            | GET    | List uploaded files   |
| `/uploads/{filename}` | DELETE | Delete uploaded file  |
| `/health`             | GET    | Health check          |

## 🖥️ Usage Examples

### Using the Web Interface

1. Open `http://localhost:8000`
2. Select input mode (Camera/Upload/Stream)
3. Configure source settings
4. Click "Start Processing"
5. View real-time pose estimation results

### Using the Standalone Demo

```bash
python topdown_demo_with_mmdet.py \
    app/pose_service/configs/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    app/pose_service/configs/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input your_video.mp4 \
    --output-root ./output \
    --device cuda:0
```

## 🔧 Development

### Running Tests

```bash
pytest app/video_service/.pytest_cache/
```

### Code Style

The project follows PEP 8 standards with additional linting:

```bash
flake8 app/
yapf --recursive --in-place app/
```

## 🚀 Performance Optimization

### GPU Acceleration

- Enable CUDA for PyTorch operations
- Use TensorRT for optimized inference (optional)
- Batch processing for improved throughput

### Memory Management

- Configurable batch sizes
- Memory pooling for GPU operations
- Efficient frame buffering

### Streaming Optimization

- WebRTC for low-latency video
- Adaptive bitrate streaming
- Frame dropping under high load

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**

- Reduce batch size in `config.py`
- Lower input resolution
- Close other GPU applications

**WebRTC Connection Failed**

- Check firewall settings
- Ensure proper network configuration
- Try different browsers

**Model Download Issues**

- Check internet connection
- Verify model URLs in configuration
- Manually download models if needed

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

## 📊 Model Information

### Supported Models

- **RTMPose**: Real-time pose estimation
- **RTMDet**: Object detection for person detection
- **Custom ONNX models**: Via ONNX runtime

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MMPose](https://github.com/open-mmlab/mmpose) - OpenMMLab pose estimation toolbox
- [MMDetection](https://github.com/open-mmlab/mmdetection) - OpenMMLab detection toolbox
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for Python
- [aiortc](https://github.com/aiortc/aiortc) - WebRTC implementation in Python


