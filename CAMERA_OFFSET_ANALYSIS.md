# Camera Worker Offset and Coordinate Interpretation Analysis

**Comparison between master and mac-refactor-attempt branches**

---

## Summary of Key Changes

The refactor changes how we interpret MediaPipe coordinates and daemon IK results, eliminating constant offsets in favor of coordinate system conversion.

---

## MediaPipe Coordinate Interpretation

### Master Branch Approach
```python
# Line 390-395 (master)
eye_center_norm = (eye_center + 1) / 2
eye_center_pixels = [
    eye_center_norm[0] * w,
    eye_center_norm[1] * h,
]
```

**Assumption:** MediaPipe returns coordinates in **[-1, +1]** range (centered at 0)

**Logic:** Convert from [-1, +1] to [0, 1] by adding 1 and dividing by 2

### Current Branch Approach
```python
# Line 392-400 (current)
# The `eye_center` value is now the correct normalized coordinate in the [0,1] range.
eye_center_norm = eye_center
eye_center_pixels = [
    eye_center_norm[0] * w,
    eye_center_norm[1] * h,
]
logger.info(f"[MEDIAPIPE COORDS] eye_center: {eye_center}, frame: {w}x{h}, pixels: {eye_center_pixels}")
```

**Assumption:** MediaPipe returns coordinates **already in [0, 1]** range

**Logic:** Use coordinates directly without transformation

**Why the change?** Testing revealed MediaPipe's `eye_center` output from `mediapipe_face_detector.py` is already normalized to [0, 1], not [-1, +1]. The master branch was effectively scaling coordinates incorrectly.

---

## Pitch Offset and Coordinate System

### Master Branch Approach
```python
# Line 101 (master) - Instance variable
self._pitch_offset = np.deg2rad(5.0)  # Offset to apply to IK pitch (positive = look more down)

# Line 413-414 (master) - Application
# Pitch: Use directly from daemon with offset (positive offset = look more down)
pitch = rotation[1] + self._pitch_offset
```

**Strategy:** Apply a **constant +5° downward offset** to daemon IK pitch

**Reasoning:** Assumes daemon's coordinate system matches robot's, but needs downward bias

### Current Branch Approach
```python
# Line 101 (current) - No offset variable

# Line 420-424 (current) - Coordinate system inversion
# 1. Interpret Pitch: Invert daemon's "positive = down" to robot's "positive = up"
pitch = -rotation[1]

# The previous upper limit of +5.0 was too restrictive. A symmetrical range is a better default.
pitch = np.clip(pitch, np.deg2rad(-25.0), np.deg2rad(5.0))
```

**Strategy:** **Invert daemon's pitch sign** to match robot's coordinate system convention

**Reasoning:**
- Daemon IK uses `positive = look down`
- Robot expects `positive = look up`
- Solution: Negate daemon's pitch result instead of adding offset

**Key Insight:** The master branch's `+5°` offset was compensating for both:
1. Coordinate system mismatch (sign inversion)
2. A desired downward bias

The current approach separates these concerns - coordinate system conversion (sign flip) is explicit, while bias can be added separately if needed.

---

## Pitch Clamping Range

### Master Branch
```python
# No explicit clamp visible in diff - likely happened elsewhere or not enforced
```

### Current Branch
```python
# Line 424 (current)
pitch = np.clip(pitch, np.deg2rad(-25.0), np.deg2rad(5.0))
```

**Range:** -25° (max up) to +5° (max down)

**Note:** Comment mentions "The previous upper limit of +5.0 was too restrictive. A symmetrical range is a better default" but the clamp still uses +5.0. This suggests the clamp may need further adjustment.

---

## Pitch Interpolation

### Master Branch Approach
```python
# Lines 446-450 (master)
# Start pitch interpolation only if change is significant
if pitch_change_deg > min_change_threshold:
    self._pitch_interpolation_target = pitch
    self._pitch_interpolation_start = self._current_interpolated_pitch
    self._pitch_interpolation_start_time = current_time
```

**Behavior:** Smoothly interpolate pitch changes over 300ms using ease-in-out

**Purpose:** Prevent jerky movements

### Current Branch Approach
```python
# Lines 453-454 (current)
# Update the pitch directly, bypassing the interpolation which causes oscillation.
self._current_interpolated_pitch = pitch
```

**Behavior:** Directly apply pitch value without interpolation

**Reasoning:** Interpolation was causing **oscillation** - the robot would overshoot and bounce back

**Why oscillation occurred:**
- Detection runs at 1Hz (every second)
- Interpolation duration is 300ms
- By the time interpolation completes, new detection arrives with different target
- Creates feedback loop: interpolate → overshoot → detect new position → interpolate back → repeat

**Solution:** Direct application lets detection frequency control movement speed naturally

---

## Yaw Handling

### Both Branches (Unchanged)
```python
# Yaw: Use raw IK result directly
yaw = rotation[2]
```

**Strategy:** Use daemon's yaw value without modification

**Why unchanged?** Yaw coordinate system appears to match between daemon and robot. Body-follow rotation compensation in `moves.py:647-660` handles alignment automatically.

---

## Debug Window Architecture

### Master Branch
```python
# Lines 638-640 (master)
# Show debug window
cv2.imshow("Face Tracking Debug", debug_frame)
cv2.waitKey(1)  # Process window events
```

**Architecture:** Debug window runs **in the same thread** as camera worker

**Display:** `cv2.imshow()` called directly in main loop

**Limitation:** May not work in headless/SSH environments

### Current Branch
```python
# Lines 641-647 (current)
# Send frame to debug window process via queue
if self.debug_frame_queue is not None:
    try:
        # Non-blocking put - drop frame if queue is full
        self.debug_frame_queue.put_nowait(debug_frame)
    except:
        pass  # Queue full, skip this frame
```

**Architecture:** Debug window runs in **separate multiprocessing process**

**Communication:** Frames sent via `multiprocessing.Queue`

**Benefits:**
- Isolates GUI from camera processing
- Non-blocking (dropped frames if queue full)
- Better for remote/SSH debugging
- Can run on different display/machine

---

## Coordinate Flow Comparison

### Master Branch Flow
```
MediaPipe detection
  ↓
eye_center (assumed [-1, +1])
  ↓
Normalize: (value + 1) / 2 → [0, 1]
  ↓
Scale to pixels: value * width/height
  ↓
Daemon IK calculation
  ↓
pitch = daemon_pitch + 5.0°
  ↓
Interpolate pitch over 300ms
  ↓
Send to robot
```

### Current Branch Flow
```
MediaPipe detection
  ↓
eye_center (already [0, 1])
  ↓
Use directly (no normalization)
  ↓
Scale to pixels: value * width/height
  ↓
Daemon IK calculation
  ↓
pitch = -daemon_pitch (coordinate inversion)
  ↓
Clamp: [-25°, +5°]
  ↓
Apply directly (no interpolation)
  ↓
Send to robot
```

---

## Testing Implications

### What to Test

1. **Face Position Accuracy**
   - Does the robot correctly track faces at different screen positions?
   - Check with logged coordinates: `[MEDIAPIPE COORDS]` in current branch

2. **Pitch Behavior**
   - Does the robot look up when face is at top of frame?
   - Does it look down when face is at bottom?
   - Is the neutral position (face centered) actually neutral (0° pitch)?

3. **Oscillation**
   - Does the robot bounce/oscillate when tracking?
   - Master branch: May oscillate due to interpolation
   - Current branch: Should be stable (direct application)

4. **Offset Bias**
   - Master: Should have slight downward bias (+5°)
   - Current: No bias (pure coordinate inversion)
   - Question: Do we want a bias? If so, add after inversion.

### Expected Behavior Differences

| Scenario | Master Branch | Current Branch |
|----------|---------------|----------------|
| Face at screen center | Look down ~5° (offset) | Look straight (0°) |
| Face at top of screen | Look up (offset by +5°) | Look up (inverted) |
| Face at bottom | Look down (offset by +5°) | Look down (inverted) |
| Tracking stability | May oscillate | Should be stable |
| Coordinate accuracy | Incorrect scaling | Correct scaling |

---

## Recommendations

### Short Term
1. **Test coordinate interpretation** - Verify MediaPipe outputs [0, 1] not [-1, +1]
2. **Validate pitch inversion** - Ensure robot looks in correct direction
3. **Monitor for oscillation** - Current branch should eliminate it
4. **Check neutral position** - Verify 0° pitch when face is centered

### Long Term
1. **Add configurable bias** - If downward bias is desired, make it explicit:
   ```python
   PITCH_BIAS_DEG = 5.0  # Degrees to look down from neutral
   pitch = -rotation[1] + np.deg2rad(PITCH_BIAS_DEG)
   ```

2. **Symmetric pitch range** - Consider changing clamp to symmetric:
   ```python
   pitch = np.clip(pitch, np.deg2rad(-25.0), np.deg2rad(25.0))
   ```

3. **Adaptive interpolation** - If direct application is too jerky, use **adaptive** interpolation that checks for oscillation

4. **Coordinate system documentation** - Document all coordinate system conventions:
   - MediaPipe: [0, 1] with origin at top-left
   - Daemon IK: positive pitch = down, radians
   - Robot motors: positive pitch = up, radians
   - Camera frame: pixels, origin at top-left

---

## Key Takeaways

✅ **Coordinate normalization was wrong** - MediaPipe already outputs [0, 1], master branch was double-normalizing

✅ **Pitch offset masked coordinate mismatch** - The +5° offset was compensating for sign inversion + adding bias

✅ **Interpolation caused oscillation** - 300ms interpolation with 1Hz detection created feedback loop

✅ **Sign inversion is the core fix** - `pitch = -rotation[1]` properly converts coordinate systems

✅ **Separation of concerns** - Current branch separates coordinate conversion (inversion) from desired bias (removed for now)

---

**Generated:** 2025-11-20
**Branch Comparison:** master vs mac-refactor-attempt (commit 4c4366a)
