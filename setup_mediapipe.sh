#!/bin/bash
# Setup script for MediaPipe tracker comparison test

echo "================================================"
echo "MediaPipe Tracker Setup"
echo "================================================"
echo

# Check if we're in the right virtual environment
if [[ "$VIRTUAL_ENV" != *"reachy/venv"* ]]; then
    echo "⚠️  WARNING: Not in reachy venv"
    echo "Run: source /home/user/reachy/venv/bin/activate"
    echo
fi

# Install MediaPipe
echo "Installing MediaPipe..."
/home/user/reachy/venv/bin/pip install mediapipe

# Check if installation succeeded
if [ $? -eq 0 ]; then
    echo "✓ MediaPipe installed successfully"
else
    echo "✗ MediaPipe installation failed"
    exit 1
fi

# Verify other dependencies
echo
echo "Checking other dependencies..."

# Check OpenCV
python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')" 2>/dev/null || echo "✗ OpenCV not found"

# Check NumPy
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" 2>/dev/null || echo "✗ NumPy not found"

# Check scipy
python3 -c "import scipy; print(f'✓ SciPy {scipy.__version__}')" 2>/dev/null || echo "✗ SciPy not found"

# Check YOLO tracker dependencies
python3 -c "from ultralytics import YOLO; print('✓ Ultralytics YOLO')" 2>/dev/null || echo "✗ Ultralytics not found"
python3 -c "from supervision import Detections; print('✓ Supervision')" 2>/dev/null || echo "✗ Supervision not found"

echo
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo
echo "To run the comparison test:"
echo "  cd /home/user/reachy/pi_reachy_deployment"
echo "  python3 test_tracker_comparison.py"
echo
echo "Options:"
echo "  --camera N      Use camera N (default: 0)"
echo "  --duration N    Run for N seconds (default: infinite)"
echo "  --width N       Display width per tracker (default: 640)"
echo
echo "Example:"
echo "  python3 test_tracker_comparison.py --duration 60"
echo
