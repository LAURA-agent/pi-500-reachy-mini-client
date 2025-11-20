# Face Tracking Pitch Inversion Bug - Mac Refactor

## Symptom

Robot looks **DOWN** when user is detected, instead of looking **AT** the user. The face tracking yaw (left/right) works correctly, but pitch (up/down) is inverted.

---

## Root Cause

The daemon IK API (`look_at_image()`) returns pitch values in a **different coordinate frame** than the robot expects.

**Daemon API Convention:**
- Positive pitch = look DOWN
- Negative pitch = look UP

**Robot Coordinate System:**
- Positive pitch = look UP
- Negative pitch = look DOWN

The original working conversation app accounts for this inversion. The mac-refactor removed it.

---

## Code Comparison

### Original Working App
**File:** `/home/user/archive/reachy_mini_conversation_app/src/reachy_mini_conversation_app/camera_worker.py:192-194`

```python
# Extract translation and rotation from the target pose directly
translation = target_pose[:3, 3]
rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

# Invert pitch: daemon API coordinate system is opposite of robot's
# Daemon: positive pitch = down, Robot needs: positive pitch = up
inverted_pitch = -rotation[1]  # ← NEGATIVE SIGN INVERTS PITCH

# Start pitch interpolation if target changed significantly (>1°)
if (self._pitch_interpolation_target is None or
    abs(inverted_pitch - self._pitch_interpolation_target) > np.deg2rad(1.0)):
    # New target - start interpolation
    self._pitch_interpolation_target = inverted_pitch
    ...
```

### Mac-Refactor (Broken)
**File:** `camera_worker.py:411-427` (both mac-refactor-attempt branch and current master)

```python
# Extract translation and rotation from the target pose directly
translation = target_pose[:3, 3]
rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

# Pitch: Use directly from daemon with offset (positive offset = look more down)
pitch = rotation[1] + self._pitch_offset  # ← NO INVERSION!

# Yaw: Use raw IK result directly
# Body-follow rotation compensation handles alignment (moves.py:647-660)
yaw = rotation[2]

# Deadband: Only start new interpolation if change is significant enough
# This prevents detection jitter from causing rapid oscillations
pitch_change_deg = abs(np.rad2deg(pitch - self._current_interpolated_pitch))
yaw_change_deg = abs(np.rad2deg(yaw - self._current_interpolated_yaw))

min_change_threshold = 1.0  # degrees - ignore changes smaller than this

# Start pitch interpolation only if change is significant
if pitch_change_deg > min_change_threshold:
    self._pitch_interpolation_target = pitch
    self._pitch_interpolation_start = self._current_interpolated_pitch
    self._pitch_interpolation_start_time = current_time
```

---

## The Fix

**Line 414** in `camera_worker.py` needs to invert the pitch:

### Option 1: Add Inversion (Keep Offset Concept)
```python
# Before (broken):
pitch = rotation[1] + self._pitch_offset

# After (fixed):
pitch = -rotation[1] + self._pitch_offset  # Invert: daemon positive=down, robot positive=up
```

### Option 2: Match Original App Exactly (Recommended)
```python
# Remove the offset entirely and just invert:
pitch = -rotation[1]  # Daemon coordinate system is inverted
```

**Note:** The original working app doesn't use a `_pitch_offset` concept at all. It just inverts the raw daemon pitch directly.

---

## Additional Differences (FYI)

The mac-refactor also changed the **neutral pitch** from the original app:

| Aspect | Original App | Mac-Refactor/Master |
|--------|-------------|---------------------|
| Neutral pitch | -10° (looking up) | 0° (looking straight) |
| Pitch offset | None | +5° (look more down) |
| Pitch inversion | YES (`-rotation[1]`) | NO (missing) |

The neutral pitch change was intentional (mentioned in previous session notes). But the missing inversion is a bug.

---

## Why This Wasn't Obvious

1. **Yaw worked correctly** - The original app uses raw `rotation[2]` for yaw (no inversion), and mac-refactor does the same. This masked the pitch issue.

2. **No error messages** - The code runs without errors. The robot just looks in the wrong direction.

3. **Coordinate frame assumptions** - The Mac-side agent assumed the daemon API returned robot-frame poses. It actually returns daemon-frame poses that need conversion.

---

## Historical Context

From `CLAUDE.md` session notes (2025-11-06):

> **2025-11-06 Session - Face Tracking Coordinate Frame Bug**
>
> User said "go read the main in the conversation app folder and check how we were handling it previously."
>
> Found the answer in original conversation app (`reachy_mini_conversation_app/src/reachy_mini_conversation_app/camera_worker.py:242`):
> ```python
> yaw = -rotation[2]    # Invert yaw because camera is mirrored
> ```
>
> **Lesson:** Check reference implementation first - Original conversation app had the answer all along.

The previous session fixed the **yaw coordinate frame** bug by referencing the original app. The same approach applies here for **pitch** - the original app shows the correct inversion.

---

## Testing the Fix

After applying the fix:
1. Enable face tracking
2. Position yourself in the camera frame
3. Move up/down relative to center
4. **Expected behavior:** Robot should look UP when you're above center, DOWN when below center
5. **Previous broken behavior:** Robot looked DOWN when above center, UP when below center

---

## Files to Modify

**Both branches need this fix:**
- `master` branch: `camera_worker.py` line 414
- `mac-refactor-attempt` branch: `camera_worker.py` line 414

Same line number, same issue, same fix.
