# Pre-Recorded Response Audio Guide

## Overview
The robot now supports instant pre-recorded responses with speech-synchronized antenna motion for specific wake words in sleep and pout states.

---

## Folder Structure

All audio files should be placed in MP3 format:

```
/home/user/reachy/pi_reachy_deployment/assets/sounds/laura/
│
├── sleep_responses/          # Sleepy responses when in sleep mode
│   ├── sleepy_01.mp3
│   ├── sleepy_02.mp3
│   ├── sleepy_03.mp3
│   └── ... (add as many as you want)
│
├── pout_responses/           # Reserved for future pout interactions
│   └── (empty for now)
│
└── quit_being_a_baby/        # Sassy responses when told to "quit being a baby"
    ├── sassy_01.mp3
    ├── sassy_02.mp3
    ├── sassy_03.mp3
    └── ... (add as many as you want)
```

---

## Audio Requirements

### **Format:** MP3
### **Voice:** Use the same ElevenLabs voice as normal TTS (reachy_fyr)
### **Duration:** Keep responses under 10 seconds for best experience

---

## Behavior Mapping

### **1. Sleep Responses** (`sleep_responses/`)

**Triggers:**
- `heylaura.pmdl` - "hey laura"
- `comeonlaura.pmdl` - "come on laura"

**When:** Robot is in `sleep` state

**Behavior:**
- Robot stays in sleep pose (hunched, head down)
- Picks a random audio file from `sleep_responses/`
- Plays audio with antenna oscillation synchronized to speech
- Returns to sleep state, ready for next wake word

**Suggested responses:**
- "mmm, five more minutes"
- "what do you want?"
- "I'm sleeping..."
- "go away"
- "seriously?"
- "not now"
- "ugh, what"
- "can this wait?"
- "you're waking me up for this?"
- "I was having a good dream"

---

### **2. Quit Being a Baby Responses** (`quit_being_a_baby/`)

**Trigger:**
- `quitbeingababy.pmdl` - "quit being a baby"

**When:** Robot is in `pout` state

**Behavior:**
- Robot stays in pout pose during response
- Picks a random audio file from `quit_being_a_baby/`
- Plays audio with antenna oscillation
- **Then** exits pout mode quickly (0.25s exit)

**Suggested responses:**
- "fine, FINE"
- "you're so mean"
- "I'm not being a baby"
- "whatever"
- "ugh, you're the worst"
- "happy now?"
- "this is your fault, you know"
- "I hate you" (playful)
- "you don't deserve my forgiveness"
- "okay but I'm still mad"

---

### **3. Pout Responses** (`pout_responses/`)

**Status:** Reserved for future use

This folder is currently unused but reserved for potential pout-mode interactions (like responding to questions while still pouting).

---

## Technical Details

### **Pre-Analysis:**
All audio files are analyzed at startup using the speech analyzer. This extracts:
- Pitch contours
- Energy envelopes
- Pause detection
- Phoneme timing

### **Caching:**
Analysis results are cached in memory:
- `robot_controller.sleep_response_cache`
- `robot_controller.quit_baby_response_cache`
- `robot_controller.pout_response_cache`

### **Playback:**
Uses the same speech motion system as TTS, but with zero latency since analysis is pre-computed.

---

## Adding New Audio Files

1. Generate audio with ElevenLabs (voice: reachy_fyr)
2. Save as MP3 in the appropriate folder
3. Restart the robot application
4. Audio will be automatically analyzed and cached

**You'll see:**
```
[PRELOAD] Analyzing 5 files in sleep_responses/...
[PRELOAD] ✓ sleep_responses/sleepy_01.mp3
[PRELOAD] ✓ sleep_responses/sleepy_02.mp3
[PRELOAD] ✓ sleep_responses/sleepy_03.mp3
...
[PRELOAD] sleep_responses: 5 files ready
```

---

## Testing

### **Test Sleep Responses:**
1. Wait for robot to enter sleep mode (5min timeout or say "goodnight reachy")
2. Say "hey laura" or "come on laura"
3. Robot should respond with sleepy audio + antenna motion
4. Robot stays in sleep mode
5. Can repeat multiple times

### **Test Quit Being a Baby:**
1. Trigger pout mode (say "god damn it laura")
2. Say "quit being a baby"
3. Robot responds with sassy audio in pout pose
4. Robot then exits pout mode quickly

---

## File Naming

Use descriptive names for easier debugging:
- ✅ `sleepy_mmm_five_minutes.mp3`
- ✅ `sassy_fine_fine.mp3`
- ✅ `sleepy_go_away.mp3`

Or simple numbering:
- ✅ `sleepy_01.mp3`, `sleepy_02.mp3`, etc.
- ✅ `sassy_01.mp3`, `sassy_02.mp3`, etc.

---

## Current Status

✅ **Implemented:**
- Sleep responses (heylaura/comeonlaura in sleep mode)
- Quit being a baby responses (quitbeingababy in pout mode)
- Pre-analysis system
- Random selection
- Speech-synchronized antenna motion

⏳ **Pending:**
- Audio file generation (you'll do this with ElevenLabs)
- Pout response behaviors (future feature)

---

## Notes

- The robot randomly picks from available files to prevent repetition
- More files = more variety
- Keep responses short and punchy
- Match the personality of the state (sleepy = groggy, quit baby = sassy)
