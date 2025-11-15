# Face Tracker Comparison Test

This directory contains a side-by-side comparison test for YOLO vs MediaPipe face tracking methods, running completely independently from the daemon and main Reachy script.

## What This Tests

### YOLO Tracker (Current)
- **Method**: YOLOv11n face detection → bounding box → face center
- **Inference**: YOLO model (via Ultralytics)
- **Output**: 2D face center in [-1, 1] coordinates, roll angle
- **Dependencies**: Ultralytics, Supervision, HuggingFace model

### MediaPipe Tracker (Proposed)
- **Method**: MediaPipe FaceMesh → 468 landmarks → SolvePnP → 3D pose
- **Inference**: MediaPipe Face Mesh model
- **Output**: Full 3D orientation (roll, pitch, yaw) in radians
- **Dependencies**: MediaPipe, scipy

## Files Created

1. **`mediapipe_head_tracker.py`** - MediaPipe-based head pose tracker
2. **`test_tracker_comparison.py`** - Side-by-side comparison test script
3. **`setup_mediapipe.sh`** - Dependency installation script
4. **`TRACKER_COMPARISON_README.md`** - This file

## Quick Start

### 1. Install MediaPipe

```bash
cd /home/user/reachy/pi_reachy_deployment
./setup_mediapipe.sh
```

Or manually:
```bash
source /home/user/reachy/venv/bin/activate
pip install mediapipe
```

### 2. Run the Comparison Test

```bash
cd /home/user/reachy/pi_reachy_deployment
python3 test_tracker_comparison.py
```

This will:
- Open camera 0 (built-in camera)
- Load both YOLO and MediaPipe trackers
- Display side-by-side comparison in real-time
- Show inference times and pose data
- Run until you press 'q'

### 3. Test Options

```bash
# Run for 60 seconds then auto-stop
python3 test_tracker_comparison.py --duration 60

# Use external USB camera (index 1)
python3 test_tracker_comparison.py --camera 1

# Larger display (800px per tracker)
python3 test_tracker_comparison.py --width 800

# Combine options
python3 test_tracker_comparison.py --camera 0 --duration 120 --width 720
```

### 4. Controls

While test is running:
- **`q`** - Quit and show summary statistics
- **`s`** - Save screenshot of current comparison

## What to Look For

### Detection Accuracy
- **Which tracker detects faces more reliably?**
- Does one fail when you turn your head to the side?
- Does one handle partial occlusion better?

### Inference Speed
- **Which is faster on the Pi?**
- Look at the FPS shown in each panel header
- Summary shows average/min/max/stddev

### Pose Quality
- **MediaPipe shows full 3D pose** (roll, pitch, yaw)
- **YOLO only shows 2D center + roll**
- Does MediaPipe's yaw estimate match your actual head rotation?
- Does the direction indicator ("LEFT 15°" / "RIGHT 10°") make sense?

### Visual Quality
- Both show face center as colored circle
- Center crosshair shows image center
- Direction indicators at bottom show head orientation

## Expected Results

### YOLO (Current System)
- **Pros**: Fast bounding box detection, proven to work
- **Cons**: Only gives 2D center, requires IK lookup for full pose
- **Speed**: Expect ~50-100ms per frame on Pi

### MediaPipe (Proposed)
- **Pros**: Full 3D pose directly, no network calls needed, 468 landmarks
- **Cons**: More complex model, might be slower
- **Speed**: Unknown on Pi - this test will tell us!

## What the Data Means

After running the test, you'll see a summary like:

```
YOLO:
  Frames processed:  600
  Faces detected:    580 (96.7%)
  Inference time:
    Average: 75.2 ms (13.3 FPS)
    Min:     62.1 ms (16.1 FPS)
    Max:     120.5 ms (8.3 FPS)
    StdDev:  12.3 ms

MEDIAPIPE:
  Frames processed:  600
  Faces detected:    595 (99.2%)
  Inference time:
    Average: 45.8 ms (21.8 FPS)
    Min:     38.2 ms (26.2 FPS)
    Max:     68.3 ms (14.6 FPS)
    StdDev:  8.1 ms
```

**What to decide:**
1. **Detection rate**: Higher % = more reliable
2. **Inference speed**: Lower ms = faster, higher FPS = better
3. **Consistency**: Lower StdDev = more predictable performance

## Next Steps Based on Results

### If MediaPipe is FASTER and MORE ACCURATE:
✅ **Replace YOLO tracker in main system**
- Eliminates network IK calls
- Higher detection frequency possible (5 Hz vs 1 Hz)
- Full 3D pose directly available

### If MediaPipe is SLOWER but MORE ACCURATE:
⚖️ **Trade-off decision needed**
- Do we value accuracy over speed?
- Can we tune detection frequency to compensate?

### If YOLO is BETTER overall:
❌ **Keep current system**
- Stick with proven YOLO approach
- Focus optimization elsewhere

## Technical Notes

### Camera Access
This test opens the camera directly via OpenCV, bypassing the daemon's `DaemonMediaWrapper`. This ensures:
- No daemon dependency
- No network overhead
- Direct comparison of tracker performance

### Coordinate Systems
- **YOLO**: Returns [-1, 1] normalized coordinates
- **MediaPipe**: Returns radians for (roll, pitch, yaw)
- Test script converts both to pixel coordinates for visualization

### Independence
This test is **completely independent** from:
- `run_pi_reachy.py` - main script
- Daemon IK endpoints - no network calls
- Movement manager - no robot control
- Display manager - just OpenCV windows

You can run this while the main system is running (using a different camera) or completely standalone.

## Troubleshooting

### "Failed to open camera"
- Check camera index: `ls /dev/video*`
- Try `--camera 1` if built-in camera is busy
- Make sure no other process is using the camera

### "MediaPipe not available"
- Run `./setup_mediapipe.sh`
- Or manually: `pip install mediapipe`

### "YOLO not available"
- YOLO dependencies already installed in main system
- Check: `pip list | grep ultralytics`

### Low FPS / High CPU
- This is expected on Pi hardware
- Close other applications
- Monitor `htop` in another terminal
- Test shows ACTUAL performance, not theoretical

## Understanding the Visual Output

### Header Bar (Top of each panel)
- **Green** = Face detected
- **Red** = No face detected
- Shows tracker name and inference time

### Face Tracking
- **Magenta circle** = Detected face center
- **Yellow ring** = Confidence indicator
- **Green crosshair** = Image center (reference)

### Pose Display (Left side)
- **Roll**: Head tilt (ear to shoulder)
- **Pitch**: Head nod (up/down)
- **Yaw**: Head turn (left/right)

### Direction Indicator (Bottom)
- **GREEN "CENTERED"** = Looking straight (±5°)
- **ORANGE "LEFT XX°"** = Looking left
- **MAGENTA "RIGHT XX°"** = Looking right

## Files You Can Delete After Testing

If you decide NOT to use MediaPipe:
```bash
rm mediapipe_head_tracker.py
rm test_tracker_comparison.py
rm setup_mediapipe.sh
rm TRACKER_COMPARISON_README.md
```

If you decide to USE MediaPipe, you'll still need to:
- Integrate it into `camera_worker.py`
- Remove IK lookup calls
- Test with full system integration

---

**Created**: 2025-11-05
**Purpose**: Evaluate MediaPipe as YOLO replacement for face tracking
**Status**: Ready for testing
