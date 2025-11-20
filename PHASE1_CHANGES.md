# Phase 1 Changes: Match Working Conversation App

## Changes Made

### 1. Neutral Pitch Changed from 0° to -10°
**Rationale:** Conversation app uses -10° neutral (looking up slightly) which is ideal for user's setup (face ~1 foot above camera)

**Locations Updated:**
- Line 102: Initial pitch interpolation value
- Line 173: Reset when tracking disabled
- Line 257: Neutral pose definition
- Line 300: Reset when tracking just disabled
- Line 533: Reset after face-lost interpolation completes

### 2. Interpolation Duration Changed from 0.3s to 2.0s
**Rationale:** Conversation app uses 2-second smooth interpolation for gradual pitch changes

**Location Updated:**
- Line 106: `self._pitch_interpolation_duration = 2.0`

### 3. Restored Pitch Interpolation Logic
**Rationale:** Direct pitch assignment was causing jerky/broken movement

**Location Updated:**
- Lines 455-459: Restored conditional interpolation start based on 1° threshold
- Removed: Direct assignment `self._current_interpolated_pitch = pitch`

### 4. Reverted to Conversation App Coordinate Transform
**Rationale:** System appears to be calibrated to the wrong coordinate math

**Location Updated:**
- Line 398: Changed from `eye_center` to `(eye_center + 1) / 2`
- Added comment explaining this is wrong but matches conversation app
- Added logging to show both raw and transformed coordinates

## What This Should Fix

✅ **Neutral position** - Robot starts looking up at -10° (ideal for face above camera)
✅ **Smooth movement** - 2s interpolation prevents jerky motion
✅ **Gradual tracking** - Interpolation logic allows smooth approach to target
✅ **Coordinate compatibility** - Uses same (wrong) math as working conversation app

## Expected Behavior

When face tracking is enabled:
- Robot should start at -10° pitch (looking up slightly)
- Smoothly interpolate to face position over 2 seconds
- Track faces using same coordinate system as conversation app
- No jerky movements or oscillation

## Testing

Run the robot with face tracking enabled and verify:
1. ✅ Starts at -10° neutral (looking up)
2. ✅ Smoothly tracks face without jerking
3. ✅ No oscillation around target position
4. ✅ Maintains face in camera view

## Next Steps

**Phase 2:** Once baseline tracking works, fix coordinate math
- Change `(eye_center + 1) / 2` back to `eye_center` (correct)
- Adjust neutral pitch if needed to compensate
- Test and recalibrate

---

**Generated:** 2025-11-20
**Status:** Phase 1 complete, ready for testing
