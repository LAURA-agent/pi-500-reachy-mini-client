# Why Archive Was Working vs Current Not Working

## Critical Discovery

**MediaPipe Returns:** `[0, 1]` normalized coordinates (from line 64-67 in mediapipe_face_detector.py)

The docstring incorrectly says `[-1, 1]` but the actual code uses MediaPipe's `relative_bounding_box` which returns `[0, 1]`.

---

## Coordinate Transformation Comparison

### Archive (Broken but Working?)
```python
# Line 328 in archive/camera_worker.py.broken
eye_center_norm = (eye_center + 1) / 2
eye_center_pixels = [
    eye_center_norm[0] * w,
    eye_center_norm[1] * h,
]
```

**Math:**
- Input: eye_center in `[0, 1]` (but code thinks it's `[-1, +1]`)
- Transformation: `(value + 1) / 2`
  - If eye_center = 0.0 → (0 + 1) / 2 = 0.5
  - If eye_center = 0.5 → (0.5 + 1) / 2 = 0.75
  - If eye_center = 1.0 → (1 + 1) / 2 = 1.0
- **Result: Converts [0, 1] to [0.5, 1.0]** ← WRONG RANGE

**Effect:** Shifts all coordinates toward bottom-right quadrant
- Center of image (0.5, 0.5) becomes (0.75, 0.75)
- Robot thinks face is in lower-right even when centered

### Current Version
```python
# Line 395 in current camera_worker.py
eye_center_norm = eye_center
eye_center_pixels = [
    eye_center_norm[0] * w,
    eye_center_norm[1] * h,
]
```

**Math:**
- Input: eye_center in `[0, 1]`
- Transformation: None
- **Result: Uses [0, 1] directly** ← CORRECT

---

## Pitch Handling Comparison

### Archive Version
```python
# Line 80 - Instance variable
self._pitch_offset = np.deg2rad(5.0)  # 5 degrees down

# Line 352 - Application
pitch = rotation[1] + self._pitch_offset
```

**Logic:**
- Takes daemon's pitch directly (daemon: positive = down)
- Adds +5° offset (look more down)
- **No coordinate system inversion**

**If daemon says "look up 10°" (rotation[1] = -10°):**
- Archive pitch = -10° + 5° = -5° (still looking up, less)

### Current Version
```python
# Line 420 - No offset variable

# Line 420-424 - Application
pitch = -rotation[1]
pitch = np.clip(pitch, np.deg2rad(-25.0), np.deg2rad(5.0))
```

**Logic:**
- Inverts daemon's pitch (daemon: positive = down, robot: positive = up)
- No offset
- Clamps to [-25°, +5°]

**If daemon says "look up 10°" (rotation[1] = -10°):**
- Current pitch = -(-10°) = +10° (looking up)

---

## Why Archive Might Have Worked Despite Wrong Coordinates

### Hypothesis 1: Compensating Errors
The coordinate shift to [0.5, 1.0] range might have accidentally compensated for:
- Incorrect camera calibration
- Wrong IK assumptions
- Physical offset in robot mounting

### Hypothesis 2: Pitch Offset Was Critical
The +5° downward bias might have been essential for:
- Compensating for camera angle
- Keeping face in camera view
- Physical robot posture

### Hypothesis 3: No Sign Inversion Needed
Maybe the daemon's coordinate system DOESN'T need inversion:
- Archive used daemon pitch directly (+ offset)
- Current inverts it
- If archive was correct, current is backwards

---

## What Changed Between Archive and Current

| Aspect | Archive | Current | Impact |
|--------|---------|---------|--------|
| **Coordinate normalization** | `(eye + 1) / 2` | Direct use | ✅ Current correct |
| **Coordinate range** | [0.5, 1.0] (wrong) | [0, 1] (correct) | ⚠️ Positions shifted |
| **Pitch sign** | No inversion | Inverted | ⚠️ May look wrong direction |
| **Pitch offset** | +5° down | None | ⚠️ May need bias |
| **Pitch interpolation** | 1.0s duration | Removed (direct) | Changed smoothness |
| **Sampling rate** | Every 1.0s | Every 1.0s | Same |

---

## What's Likely Wrong in Current Version

### Problem 1: Pitch Sign May Be Correct in Archive
**Test:** If current version looks UP when face is BELOW center:
- Daemon might already return correct sign for robot
- Current inversion (-rotation[1]) is backwards
- **Fix:** Remove negation, use `pitch = rotation[1]` like archive

### Problem 2: Missing Pitch Offset
**Symptom:** Robot looks too high/low even when centered
- Archive had +5° downward bias
- Current has none
- **Fix:** Add back offset after sign is correct

### Problem 3: Coordinate Shift Broke Calibration
**Symptom:** Robot looks in wrong position even with correct math
- Archive's [0.5, 1.0] shift might have compensated for camera/IK calibration
- Current's correct [0, 1] exposes calibration issues
- **Fix:** Recalibrate camera or add position offset

---

## Recommended Testing Sequence

### Step 1: Test Pitch Sign
```python
# In camera_worker.py line 420, try archive approach:
pitch = rotation[1]  # No inversion
```
- Put face at TOP of frame → robot should look UP
- Put face at BOTTOM of frame → robot should look DOWN
- If backwards, try current inversion again

### Step 2: Add Pitch Offset
```python
# After determining correct sign, add offset:
pitch = rotation[1] + np.deg2rad(5.0)  # Archive's offset
```
- Check if robot looks at centered face with neutral pitch

### Step 3: Verify Coordinate Range
```python
# Add logging to see what daemon receives:
logger.info(f"Pixel coords sent to IK: u={u_pixel}, v={v_pixel}, frame={w}x{h}")
```
- Face at center (0.5, 0.5) should send pixels near (640, 360) for 1280x720
- If offset, coordinate math is still wrong

### Step 4: Test Coordinate Transformation
```python
# Try archive's transformation temporarily:
eye_center_norm = (eye_center + 1) / 2
```
- If this makes it work, there's a hidden calibration issue

---

## Quick Diagnostic Script

Add this to camera_worker.py after line 400:

```python
logger.info(f"""
[FACE TRACKING DEBUG]
Raw eye_center from MediaPipe: {eye_center}
Normalized (current): {eye_center_norm}
Pixel coordinates: u={u_pixel}, v={v_pixel}
Frame size: {w}x{h}
Daemon IK rotation (raw): roll={np.rad2deg(rotation[0]):.1f}° pitch={np.rad2deg(rotation[1]):.1f}° yaw={np.rad2deg(rotation[2]):.1f}°
Final pitch sent: {np.rad2deg(pitch):.1f}°
""")
```

This will show exactly what's being sent to the robot.

---

## Most Likely Fix

Based on the comparison, I suspect:

1. **Pitch sign doesn't need inversion** - Daemon already returns correct sign
2. **Pitch offset is critical** - +5° downward bias compensates for camera angle
3. **Coordinate math is now correct** - But may expose calibration issues

**Try this in camera_worker.py line 420:**
```python
# Use daemon pitch directly with offset (like archive)
pitch = rotation[1] + np.deg2rad(5.0)
pitch = np.clip(pitch, np.deg2rad(-25.0), np.deg2rad(25.0))
```

This reverts to the archive's pitch handling while keeping correct coordinate normalization.

---

**Generated:** 2025-11-20
**Purpose:** Debug why current version doesn't work vs archive
