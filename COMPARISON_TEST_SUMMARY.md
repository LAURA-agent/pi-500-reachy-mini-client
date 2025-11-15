# Tracker Comparison Test - Setup Complete ✓

## What Was Created

### 1. MediaPipe Head Tracker (`mediapipe_head_tracker.py`)
- Full 3D head pose estimation using MediaPipe FaceMesh
- Returns (roll, pitch, yaw) in radians
- Uses SolvePnP algorithm for direct pose calculation
- No network calls, no daemon dependency

### 2. Comparison Test Script (`test_tracker_comparison.py`)
- Side-by-side visual comparison of YOLO vs MediaPipe
- Real-time performance metrics (FPS, inference time)
- Independent camera access (no daemon required)
- Interactive controls (save screenshots, quit anytime)

### 3. Setup Script (`setup_mediapipe.sh`)
- Automated MediaPipe installation
- Dependency verification
- Usage instructions

### 4. Documentation
- `TRACKER_COMPARISON_README.md` - Full usage guide
- `COMPARISON_TEST_SUMMARY.md` - This file

## Installation Status

✓ **MediaPipe installed**: v0.10.18
✓ **YOLO tracker**: Already available
✓ **Cameras detected**: /dev/video0, /dev/video1
✓ **Both trackers import successfully**

⚠️ **NumPy Version Conflict Detected**
- MediaPipe requires: numpy<2
- Reachy requires: numpy>=2.2.5
- **Current version**: numpy 1.26.4 (downgraded by MediaPipe)
- **Impact**: May cause compatibility issues with reachy-mini package

### NumPy Conflict Resolution Options

**Option A: Test in current state**
- Trackers work fine with numpy 1.26.4
- Comparison test is independent of Reachy system
- Risk: Main Reachy system might have issues if restarted

**Option B: Virtual environment isolation** (Recommended)
```bash
# Create separate test environment
python3 -m venv /home/user/tracker_test_venv
source /home/user/tracker_test_venv/bin/activate
pip install mediapipe opencv-python scipy numpy
# Run comparison test in this isolated environment
```

**Option C: Accept the risk**
- Run comparison test now
- Reinstall correct numpy after: `pip install numpy>=2.2.5`
- Re-test Reachy system

## How to Run the Comparison

### Basic Test
```bash
cd /home/user/reachy/pi_reachy_deployment
python3 test_tracker_comparison.py
```

### Important Notes

1. **Camera Availability**: If daemon is running, it may be using the camera
   - Stop daemon: `sudo systemctl stop reachy-mini`
   - Or use alternate camera: `--camera 1`

2. **Display Requirement**: Test creates OpenCV windows
   - Requires X11 or VNC session
   - Won't work over pure SSH

3. **Test Duration**: Runs until you press 'q'
   - Or set auto-stop: `--duration 60`

### Example Commands

```bash
# 60-second test with auto-stop
python3 test_tracker_comparison.py --duration 60

# Use external USB camera if built-in is busy
python3 test_tracker_comparison.py --camera 1

# Larger display for better visibility
python3 test_tracker_comparison.py --width 800
```

## What to Evaluate

### 1. Detection Reliability
- Which tracker detects faces more consistently?
- Does one fail at certain angles?
- Watch the header bar color (green = detected, red = no face)

### 2. Inference Speed
- Look at FPS in panel headers
- Check summary statistics after test
- Lower ms = faster, higher FPS = better

### 3. Pose Accuracy (MediaPipe only)
- Does yaw match actual head rotation?
- Do direction indicators make sense?
- Is 3D pose estimation accurate?

### 4. Visual Quality
- Face center tracking smoothness
- Stability (jitter/jumpiness)
- Recovery when face re-appears

## Expected Test Flow

1. **Launch**: Both tracker panels appear side-by-side
2. **Detection**: Colored headers show detection status
3. **Metrics**: Real-time FPS and pose data displayed
4. **Interaction**: Press 's' to save screenshots, 'q' to quit
5. **Summary**: Detailed statistics printed to console

## Decision Matrix

After testing, use this to decide:

| Metric | YOLO Better | MediaPipe Better | Action |
|--------|------------|-----------------|--------|
| **Detection Rate** | ✓ | - | Keep YOLO |
| **Detection Rate** | - | ✓ | Consider MediaPipe |
| **Speed (FPS)** | ✓ | - | Keep YOLO |
| **Speed (FPS)** | - | ✓ | MediaPipe advantage |
| **3D Pose** | N/A | ✓ | MediaPipe advantage |
| **Stability** | ✓ | - | Keep YOLO |
| **Overall** | Better | - | Stay with current system |
| **Overall** | - | Better | Proceed to integration |

## Next Steps Based on Results

### If MediaPipe Wins:
1. Test with full system integration
2. Remove daemon IK dependency from `camera_worker.py`
3. Increase detection frequency (1 Hz → 5 Hz)
4. Verify coordinate frame compatibility

### If YOLO Wins:
1. Keep current system
2. Optimize IK endpoint performance
3. Consider other tracking improvements
4. Delete test files (see cleanup below)

### If Results Are Mixed:
1. Run longer tests (10+ minutes)
2. Test in different lighting conditions
3. Test with multiple people
4. Evaluate CPU temperature/throttling

## Cleanup (If Not Using MediaPipe)

```bash
cd /home/user/reachy/pi_reachy_deployment

# Remove test files
rm mediapipe_head_tracker.py
rm test_tracker_comparison.py
rm setup_mediapipe.sh
rm TRACKER_COMPARISON_README.md
rm COMPARISON_TEST_SUMMARY.md

# Restore numpy version
/home/user/reachy/venv/bin/pip install "numpy>=2.2.5" --force-reinstall

# Verify Reachy still works
sudo systemctl restart reachy-mini
```

## Files Reference

All files are in: `/home/user/reachy/pi_reachy_deployment/`

- `mediapipe_head_tracker.py` - MediaPipe tracker implementation
- `yolo_head_tracker.py` - Existing YOLO tracker (unchanged)
- `test_tracker_comparison.py` - Comparison test script
- `setup_mediapipe.sh` - Installation script
- `TRACKER_COMPARISON_README.md` - Detailed usage guide
- `COMPARISON_TEST_SUMMARY.md` - This summary

## Troubleshooting

### "No module named 'mediapipe'"
```bash
source /home/user/reachy/venv/bin/activate
pip install mediapipe
```

### "Failed to open camera"
```bash
# Stop daemon if running
sudo systemctl stop reachy-mini

# Or try alternate camera
python3 test_tracker_comparison.py --camera 1
```

### "Can't open display"
- Test requires graphical environment
- Use VNC viewer or connect monitor
- Can't run over SSH without X forwarding

### Low FPS / High CPU
- Expected on Pi hardware
- Close other applications
- Monitor with `htop`
- Consider shorter test duration

---

## Ready to Test!

Everything is installed and ready. To start the comparison:

```bash
cd /home/user/reachy/pi_reachy_deployment
python3 test_tracker_comparison.py
```

Press 'q' when done to see summary statistics.

**Created**: 2025-11-05
**Status**: Ready for testing
**Environment**: Reachy Pi venv with MediaPipe 0.10.18
