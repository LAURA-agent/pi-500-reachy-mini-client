# Face Tracking Fix - Final Summary

## Problem
Face tracking was causing grainy, jerky head movements with oscillation. The robot would snap to targets instead of smoothly tracking faces.

## Root Causes Identified

### 1. Wrong Coordinate Transform
**Issue:** Conversation app used `(eye_center + 1) / 2` which shifted all coordinates
**Reality:** MediaPipe returns [0,1] coordinates, not [-1,+1]
**Fix:** Use coordinates directly without transformation

### 2. Wrong Neutral Pitch
**Issue:** Used 0° neutral which was too low for user's setup (face 1ft above camera)
**Fix:** Changed to -10° (looking up slightly) to match conversation app

### 3. Missing Interpolation
**Issue:** Detection at 10Hz was canceling 2s interpolation before completion
**Fix:** Reduced detection frequency to 1Hz, allowing 2s interpolation to complete

### 4. Race Condition (THE BIG ONE)
**Issue:** 100Hz control loop sent raw face tracking offsets that fought with breathing motion
**Result:** Grainy side-to-side movement, jittery tracking
**Fix:** Added low-pass filter (EMA with α=0.92) to smooth offsets before application

## Implementation Summary

### Phase 1: Match Conversation App Baseline
- Set neutral pitch to -10° throughout codebase
- Changed interpolation duration from 0.3s to 2.0s
- Restored pitch interpolation logic (removed direct assignment)
- Temporarily used conversation app's coordinate transform

### Phase 2: Fix Coordinate Math
- Removed incorrect `(eye_center + 1) / 2` transform
- Use MediaPipe coordinates directly (already [0,1])
- Fixed docstring to reflect actual coordinate range
- Added coordinate logging for debugging

### Phase 3: Fix Detection Frequency
- Changed from 10Hz (every 0.1s) to 1Hz (every 1.0s)
- Allows 2s interpolation to mostly complete before new detection
- Prevents constant target changes that cause oscillation

### Phase 4: Add Low-Pass Filter (Critical Fix)
- Added exponential moving average filter to face tracking offsets
- Filter runs at 100Hz in MovementManager._update_face_tracking()
- Smoothing factor α=0.92 (tunable: 0.0=no filter, 0.95=very smooth)
- Prevents rapid offset changes from fighting with breathing motion

## Final Configuration

```python
# camera_worker.py
self._current_interpolated_pitch = np.deg2rad(-10.0)  # Neutral: look up
self._pitch_interpolation_duration = 2.0  # Smooth 2s interpolation
should_sample = (current_time - self._last_sample_time) >= 1.0  # 1Hz detection
eye_center_norm = eye_center  # Direct use of [0,1] coordinates
pitch = -rotation[1]  # Invert daemon coordinate system

# moves.py
self._face_tracking_smoothing = 0.92  # Low-pass filter coefficient
# Applied in _update_face_tracking() at 100Hz
smoothed = 0.92 * smoothed_prev + 0.08 * raw_value
```

## Key Insights

1. **Interpolation alone isn't enough** - Even with 2s smooth interpolation, the 100Hz control loop sends those intermediate values directly to motors, fighting with breathing

2. **Coordinate systems matter** - MediaPipe's [0,1] range was being incorrectly transformed, causing offset tracking

3. **Detection frequency must match interpolation** - 10Hz detection with 2s interpolation = 20 target changes per interpolation = oscillation

4. **Face tracking is an offset, not a target** - It must be smoothed before being combined with breathing motion at 100Hz

## Performance Characteristics

- **Detection:** 1Hz (every 1.0 second)
- **Interpolation:** 2.0 seconds per target change
- **Control loop:** 100Hz (every 0.01 second)
- **Low-pass filter:** α=0.92 exponential moving average
- **Response time:** ~1-2 seconds to reach new face position
- **Smoothness:** No jitter, works harmoniously with breathing

## Testing Results

✅ Face tracking starts at -10° neutral position
✅ Smoothly interpolates to face position over 2 seconds
✅ No oscillation around target
✅ No grainy/jerky side-to-side movement
✅ Breathing and face tracking work together smoothly
✅ Maintains face in camera view consistently

---

**Generated:** 2025-11-20
**Status:** Complete and working
**Branch:** mac-refactor-attempt
**Commits:** 67beaa7, 2f01c41, 3a3562d, 93d55e9
