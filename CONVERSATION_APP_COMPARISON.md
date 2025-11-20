# Conversation App vs Current Implementation

## Key Discovery: Three Different Approaches

### 1. Archive (Broken camera_worker.py)
- Coordinate: `(eye_center + 1) / 2` ← WRONG
- Pitch: `rotation[1] + 5°` ← NO inversion, +5° offset
- Neutral: 0°
- Interpolation: 1.0s

### 2. Reachy Mini Conversation App (Official Pollen)
- Coordinate: `(eye_center + 1) / 2` ← WRONG (same as archive)
- Pitch: `-rotation[1]` ← INVERTED
- Neutral: **-10°** (looking UP)
- Interpolation: **2.0s** duration

### 3. Current Implementation
- Coordinate: Direct use ← CORRECT
- Pitch: `-rotation[1]` ← INVERTED (same as conversation app)
- Neutral: 0°
- Interpolation: **REMOVED**

---

## Critical Code from Conversation App

```python
# Line 76 - Neutral pitch is -10° (looking UP)
self._current_interpolated_pitch = np.deg2rad(-10.0)

# Line 147 - Neutral pose definition
neutral_pose = np.eye(4, dtype=np.float32)
neutral_pose[2, 3] = 0.015  # Raise head by 1.5cm
neutral_rotation = R.from_euler("xyz", [0, np.deg2rad(-10.0), 0])  # -10° pitch

# Line 191 - Wrong coordinate transformation (but works?)
eye_center_norm = (eye_center + 1) / 2

# Line 214 - Pitch inversion
inverted_pitch = -rotation[1]

# Line 260-274 - 2-second interpolation
self._pitch_interpolation_duration = 2.0
# ... linear interpolation code
```

---

## Why Conversation App Works Despite Wrong Coordinates

### The Wrong Math
`(eye_center + 1) / 2` converts:
- Input [0, 1] → Output [0.5, 1.0]
- Center (0.5, 0.5) becomes (0.75, 0.75)
- Robot thinks face is in lower-right quadrant even when centered

### Possible Compensation
1. **Pitch offset of -10°** might compensate for coordinate shift
2. **2-second interpolation** smooths out errors
3. **Camera/IK calibration** might be tuned to wrong coordinates

---

## The Real Problem: Three Incompatible Changes

Your current version made **three changes simultaneously**:

1. ✅ **Fixed coordinate math** - Correct, but breaks calibration
2. ⚠️ **Changed neutral from -10° to 0°** - May need to keep -10°
3. ❌ **Removed interpolation** - Causes jerky movement

**The conversation app that works has:**
- Wrong coordinates (but system is calibrated to it)
- -10° neutral (looking up)
- 2-second smooth interpolation

---

## Recommended Fix: Match Conversation App Exactly

### Step 1: Revert to Conversation App's Approach

```python
# In camera_worker.py

# Line 101 - Set neutral to -10° like conversation app
self._current_interpolated_pitch = np.deg2rad(-10.0)  # Look up slightly

# Line 204 - Use neutral -10° like conversation app
neutral_rotation = R.from_euler("xyz", [0, np.deg2rad(-10.0), 0])

# Line 395 - TEMPORARILY use wrong coordinates like conversation app
eye_center_norm = (eye_center + 1) / 2  # WRONG but system is calibrated to it

# Line 420 - Keep inversion like conversation app
pitch = -rotation[1]  # Invert daemon pitch

# Line 453 - RESTORE interpolation like conversation app
# Change from:
self._current_interpolated_pitch = pitch
# Back to:
if pitch_change_deg > min_change_threshold:
    self._pitch_interpolation_target = pitch
    self._pitch_interpolation_start = self._current_interpolated_pitch
    self._pitch_interpolation_start_time = current_time
```

### Step 2: Change Interpolation Duration

```python
# Line 106 - Change from 0.3s to 2.0s like conversation app
self._pitch_interpolation_duration = 2.0  # Match conversation app
```

---

## Why Each Change Matters

### 1. Neutral at -10° (Looking Up)
- Conversation app starts looking UP slightly
- Archive started at 0° (straight ahead)
- **Current uses 0°** ← May be too low

### 2. 2-Second Interpolation
- Conversation app: 2.0s smooth ramp
- Archive: 1.0s ramp
- **Current: NONE** ← Very jerky

### 3. Wrong Coordinate Transform
- Both archive AND conversation app use wrong math
- System might be calibrated expecting wrong coordinates
- **Current fixed it** ← But broke calibration

---

## Test Plan

### Test 1: Match Conversation App Exactly
```python
# Use all three conversation app settings:
1. eye_center_norm = (eye_center + 1) / 2  # Wrong transform
2. neutral = -10°
3. interpolation_duration = 2.0s
4. pitch = -rotation[1]  # Inverted
```

**Expected:** Should work exactly like conversation app

### Test 2: Fix Coordinates, Keep Rest
```python
# Only fix coordinate math:
1. eye_center_norm = eye_center  # CORRECT
2. neutral = -10°  # Keep conversation app value
3. interpolation_duration = 2.0s  # Keep conversation app value
4. pitch = -rotation[1]  # Keep inversion
```

**Expected:** May need to adjust neutral pitch to compensate for coordinate fix

### Test 3: Calibrate New Neutral
```python
# With correct coordinates, find right neutral:
1. eye_center_norm = eye_center  # CORRECT
2. neutral = -15° or -20°?  # Experiment
3. interpolation_duration = 2.0s
4. pitch = -rotation[1]
```

**Expected:** Find neutral that keeps face in view with correct coordinates

---

## Immediate Action

Replace these lines in camera_worker.py:

```python
# Line 101 - Change neutral
self._current_interpolated_pitch = np.deg2rad(-10.0)  # was 0.0

# Line 106 - Restore interpolation duration
self._pitch_interpolation_duration = 2.0  # was 0.3

# Line 204 - Change neutral pose
neutral_rotation = R.from_euler("xyz", [0, np.deg2rad(-10.0), 0])  # was 0.0

# Line 395 - TEMPORARILY revert to wrong transform
eye_center_norm = (eye_center + 1) / 2  # was: eye_center

# Line 453-454 - RESTORE interpolation (remove direct assignment)
# DELETE these lines:
# self._current_interpolated_pitch = pitch

# ADD back interpolation logic:
if pitch_change_deg > min_change_threshold:
    self._pitch_interpolation_target = pitch
    self._pitch_interpolation_start = self._current_interpolated_pitch
    self._pitch_interpolation_start_time = current_time
```

This will make your current version behave **exactly like the working conversation app**.

Then you can incrementally fix the coordinate math and recalibrate.

---

## Summary

**Why current version doesn't work:**
1. ❌ Removed interpolation → jerky movement
2. ❌ Changed neutral from -10° to 0° → wrong default position
3. ⚠️ Fixed coordinates → system not calibrated for correct math

**Why conversation app works:**
1. ✅ Uses 2s interpolation → smooth
2. ✅ Uses -10° neutral → keeps face in view
3. ⚠️ Uses wrong coordinates → but system is calibrated to them

**The fix:**
Match conversation app exactly, THEN incrementally fix coordinate math.

---

**Generated:** 2025-11-20
