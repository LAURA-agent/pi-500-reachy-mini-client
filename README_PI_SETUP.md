# Reachy Mini Pi Control - Setup Guide

## Overview

This package enables complete Reachy Mini control from a Raspberry Pi 500, including:
- Breathing animations (continuous idle behavior)
- Face tracking with YOLO detection
- Mood-based emotion movements synchronized with TTS
- MCP integration with Mac server for LLM processing

## Prerequisites

### Hardware
- Raspberry Pi 500 (or Pi 4/5 with sufficient RAM)
- Reachy Mini robot (physical or simulator)
- USB camera (if using face tracking)
- Microphone for wake word detection
- Network connection to Mac (for MCP server)

### Software
- Raspberry Pi OS (64-bit recommended)
- Python 3.10 or newer
- ~2GB free disk space (for models and dependencies)

## Installation

### 1. Extract Files

```bash
# Extract deployment package
cd ~/
unzip pi_reachy_deployment.zip
cd pi_reachy_deployment
```

### 2. Create Virtual Environment

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install Python packages
pip install -r requirements_pi_reachy.txt

# Download YOLO model (for face tracking)
python3 << EOF
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt", local_dir="./models")
EOF
```

### 4. Start Reachy Daemon

**Simulator Mode (for testing without physical robot):**
```bash
# Requires X display or VNC
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100
```

**Physical Robot Mode:**
```bash
# For actual Reachy Mini hardware
mjpython -m reachy_mini.daemon.app.main --fastapi-port 8100
```

**Note:** Keep this running in a separate terminal/tmux session.

### 5. Configure MCP Connection

Edit the MCP server URL in `run_pi_reachy.py` if your Mac has a different IP:

```python
# Line ~35
SERVER_URL = "http://192.168.0.124:8765/mcp"  # Update to your Mac IP
```

### 6. Test Daemon Connection

```bash
# Verify daemon is accessible
curl http://localhost:8100/api/daemon/status
```

Expected output: `{"simulation_enabled": true, ...}`

## Running the System

### Basic Startup

```bash
# Activate venv
source venv/bin/activate

# Run integrated client
python3 run_pi_reachy.py
```

### Expected Startup Sequence

1. **Boot** - Daemon connection established
2. **Idle** - Breathing animation starts after 0.3s
3. **Face Tracking** - Head follows detected faces (during idle/breathing only)
4. **Wake Word Ready** - "Hey Laura" or keyboard trigger

### Testing Checklist

- [ ] Daemon running on localhost:8100
- [ ] Breathing animation visible
- [ ] Face tracking follows face (during idle)
- [ ] Wake word triggers listening state
- [ ] Transcription sent to Mac MCP
- [ ] Mood movement plays during TTS
- [ ] Face tracking disabled during listening/thinking/speaking
- [ ] Returns to idle/breathing after response

## Configuration

### Camera Source

By default, uses Reachy's camera via MediaManager. To use a separate USB webcam:

1. Modify `camera_worker.py` initialization
2. Update camera index or device path
3. Adjust IK calculations for webcam perspective

### Face Tracking Sensitivity

Edit `yolo_head_tracker.py`:
```python
CONFIDENCE_THRESHOLD = 0.5  # Lower = more sensitive (default)
```

### Breathing Parameters

Edit `moves.py` BreathingMove initialization:
```python
sway_amplitude=0.008,  # 8mm Y sway (default)
roll_amplitude=np.deg2rad(6.0),  # 6° roll (default)
breathing_frequency=0.2,  # 0.2 Hz (5s cycle, default)
```

### TTS Server

If using local Pi TTS instead of Mac server, update TTSHandler configuration:
```python
TTS_SERVER_URL = "http://localhost:5000"  # Local instead of Mac IP
```

## Troubleshooting

### Daemon Won't Connect

```bash
# Check if daemon is running
ps aux | grep mjpython

# Check port availability
sudo lsof -i :8100

# Test direct connection
curl http://localhost:8100/api/daemon/status
```

### No Face Tracking

```bash
# Verify camera access
python3 -c "import cv2; print(cv2.VideoCapture(0).read())"

# Check YOLO model
ls -lh models/model.pt

# Enable debug window
# In camera_worker.py: debug_window=True
```

### MCP Connection Failed

```bash
# Verify Mac MCP server is running
curl http://192.168.0.124:8765/mcp/health

# Check network connectivity
ping 192.168.0.124

# Verify firewall allows port 8765
```

### Breathing Not Starting

- Check daemon status (must be running)
- Verify no active moves blocking breathing
- Check idle_inactivity_delay (default 0.3s)
- Review logs for MovementManager errors

### Mood Movements Not Playing

- Verify mood marker in MCP response (`<!-- MOOD: celebratory -->`)
- Check external_control flag not stuck
- Confirm emotion names exist in dataset
- Test daemon move endpoint directly:
  ```bash
  curl -X POST http://localhost:8100/api/move/play/recorded-move-dataset/pollen-robotics/reachy-mini-emotions-library/amazed1
  ```

## Performance Optimization

### CPU Usage

Monitor with `htop`:
- MovementManager: ~5-10% CPU (100Hz loop)
- CameraWorker: ~10-20% CPU (30Hz + YOLO)
- Target: <50% total CPU usage

If CPU is high:
- Reduce MovementManager frequency to 50Hz
- Reduce CameraWorker polling to 15Hz
- Use smaller YOLO model (YOLOv8n instead of v8s)

### Memory Usage

Monitor with `free -h`:
- Base: ~500MB
- YOLO model: ~200MB
- Target: <2GB total

### Network Latency

Test MCP round-trip time:
```bash
time curl -X POST http://192.168.0.124:8765/mcp/test
```

Target: <200ms

## File Structure

```
pi_reachy_deployment/
├── run_pi_reachy.py        # Main integrated client
├── state_tracker.py        # Lightweight state management
├── mood_extractor.py       # Mood→emotion mapping with move duration
├── moves.py                # MovementManager + BreathingMove
├── camera_worker.py        # Face detection + tracking
├── daemon_client.py        # Reachy daemon REST API wrapper
├── head_wobbler.py         # Speech sway coordination
├── yolo_head_tracker.py    # YOLO face detection
├── requirements_pi_reachy.txt
└── README_PI_SETUP.md      # This file
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Raspberry Pi 500                 │
├─────────────────────────────────────────┤
│  Main Async Loop (asyncio)              │
│  - MCP client                            │
│  - StateTracker                          │
│  - Wake word detection                   │
├─────────────────────────────────────────┤
│  Thread 1: MovementManager (100Hz)      │
│  - Breathing                             │
│  - Mood movements                        │
│  - Pose composition                      │
├─────────────────────────────────────────┤
│  Thread 2: CameraWorker (30Hz)          │
│  - YOLO face detection                   │
│  - Tracking offsets                      │
├─────────────────────────────────────────┤
│  Thread 3: HeadWobbler                   │
│  - Speech sway                           │
├─────────────────────────────────────────┤
│  Reachy Daemon (localhost:8100)         │
│  - MuJoCo simulator OR hardware drivers │
└─────────────────────────────────────────┘
           ↓
    Network to Mac MCP Server
    (LLM processing only)
```

## Support

For issues:
1. Check troubleshooting section above
2. Review logs in terminal output
3. Test each component individually
4. Verify daemon is responding correctly

## License

Apache 2.0 (aligns with Reachy Mini SDK)
