# Pout Mode Implementation Summary

## Status: 11/12 Tasks Complete ✅

All core components implemented and ready for testing. Only wake word loop integration remains.

---

## ✅ Completed Components

### 1. Movement System (`moves.py`) - ~500 new lines

**PoutPoseMove** (lines 775-879)
- Maintains SLEEP_HEAD_POSE from SDK
- Discrete body rotations: 0°, ±45°, ±90°
- Frozen breathing (no sway/roll)
- Smooth 2s interpolation to pout pose
- `rotate_to(angle)` method for dynamic rotation

**AntennaTwitchMove** (lines 882-960)
- Pattern-based antenna animations
- `frustration_twitch`: Random 1-1.5s flicks, ±3° (Sailor Moon style)
- `angry_swing`: ±5° larger movements
- `nervous_flutter`: ±1.5° quick small movements
- Base antenna position configurable

**SequenceMove Base Class** (lines 963-1086)
- Multi-step choreography framework
- `add_step(duration, target_pose, target_antennas, target_body_yaw)`
- Automatic interpolation between steps
- Non-interruptible guarantee

**Pout Exit Sequences** (lines 1093-1250)

`create_pout_exit_lunge_sequence()` - "quit being a baby laura"
1. Exit pout → neutral (2s)
2. Tilt head down -15°, point at user (1s)
3. Lunge forward 3cm (0.5s)
4. Hold aggressive posture (1s)
5. Return to neutral (1s)
**Total: 5.5s**

`create_pout_exit_gentle_sequence()` - "i'm sorry laura"
1. Slow exit → neutral (3s, reluctant)
2. Gentle tilt toward user +5° (1s)
3. Return to neutral (1s)
**Total: 5s**

---

### 2. Wake Word Configuration System

**config/wake_words.yml**
```yaml
wake_words:
  - phrase: "quit being a baby laura"
    context_injection: "User called you out for pouting..."
    movement_sequence: "pout_exit_lunge"
    active_in_states: ["pout"]
    scene_manager_override: true

  - phrase: "i'm sorry laura"
    movement_sequence: "pout_exit_gentle"
    active_in_states: ["pout"]

  # + 6 more behaviors defined
```

**wake_word_manager.py** - ~250 lines
- `WakeWordManager` class
- `get_active_wake_words(state, scene_manager_active)`
- `inject_context(phrase)` → system instruction append
- `get_movement_sequence_name(phrase)`
- `requires_scene_manager_approval(phrase)`

---

### 3. Display Integration

**display/display_manager.py**
- Added 'pout' state to states dictionary (lines 102, 330)
- Pout sprites copied to `/home/user/more transfer/assets/images/laura rp client images/pout/`
- Image rotation enabled for pout state (lines 258, 280)
- 3 pout sprite images: pout01.png, pout02.png, pout03.png

---

### 4. LLM Tool Functions

**tools/pout_tools.py** - ~200 lines

Functions:
- `enter_pout_mode()` → Trigger pout state
- `pout_rotate_to(angle)` → Body rotation: -90, -45, 0, 45, 90
- `play_antenna_pattern(pattern)` → Emotional animations
- `exit_pout_mode(reason)` → Return to normal

MCP Tool Metadata: `POUT_TOOLS` array ready for server registration

---

### 5. State Handlers (`run_pi_reachy.py`)

**Initialization** (lines 192-196):
```python
self._in_pout_mode = False
self._pout_move = None
self._pout_antenna_twitch = None
self.wake_word_manager = WakeWordManager()
```

**Methods Added** (lines 267-414):
- `enter_pout_mode(initial_body_yaw_deg=0.0)`
- `exit_pout_mode(wake_word_phrase=None)`
- `pout_rotate_to(angle_deg)`
- `pout_play_antenna_pattern(pattern="frustration_twitch")`

**Features**:
- Automatic pose capture
- Safe fallbacks on errors
- Face detection integration for targeted sequences
- Movement queue management
- Display state synchronization

---

## 🔨 Remaining Work

### Task 11: Wake Word Loop Integration

**Location**: `run_pi_reachy.py` line 700-740 (`run_main_loop()`)

**Required Changes**:

1. **Add 'pout' to wake-responsive states** (line 701):
```python
# Before:
if current_state in ['sleep', 'idle', 'code']:

# After:
if current_state in ['sleep', 'idle', 'code', 'pout']:
```

2. **Add contextual wake word handling** (after line 730, before persona switching):
```python
# Check for contextual wake words (pout-specific)
if current_state == 'pout' and 'wakeword' in wake_event_source:
    model_name = wake_event_source.split('(')[1].rstrip(')')

    # Get active wake words for pout state
    active_wake_words = self.wake_word_manager.get_active_wake_words('pout', scene_manager_active=False)

    # Map model_name to wake word phrase (TODO: need mapping logic)
    detected_phrase = None  # Map snowboy model to phrase

    if detected_phrase in active_wake_words:
        print(f"[POUT] Contextual wake word detected: {detected_phrase}")

        # Inject context to conversation
        context = self.wake_word_manager.inject_context(detected_phrase)
        # TODO: Append context to next LLM request

        # Trigger movement sequence
        self.exit_pout_mode(wake_word_phrase=detected_phrase)

        # Continue with normal wake flow (listening state, etc.)
```

3. **Snowboy Model → Phrase Mapping**:

Need to create mapping from `.pmdl` model names to wake word phrases:
- `quit_being_a_baby_laura.pmdl` → `"quit being a baby laura"`
- `im_sorry_laura.pmdl` → `"i'm sorry laura"`
- `hey_laura.pmdl` → `"hey laura"`
- etc.

**Options**:
a) Add mapping dict to wake_words.yml
b) Convention: model filename = phrase with underscores
c) Train new Snowboy models for each wake word phrase

---

### Task 12: Laura Agent Updates

**Location**: `scene_manager/agents/laura_agent.py`

**Required Updates**:

1. **Add pout state awareness to system instructions**:
```
You can enter pout mode when frustrated or offended.
While pouting, you can rotate away, swing antennas, and respond curtly.
Exit pout when user apologizes or you decide to forgive them.
```

2. **Add tool availability context**:
```
Available pout tools:
- enter_pout_mode() - Go into hiding/pouting
- pout_rotate_to(angle) - Turn away dramatically
- play_antenna_pattern(pattern) - Express frustration
- exit_pout_mode(reason) - Forgive and return
```

3. **Add decision logic for scene_manager mode**:
When `scene_manager_override: true` wake word detected:
- Director agent queries laura_agent: "User said '{phrase}'. Exit pout?"
- Laura decides based on:
  - Quality of apology
  - How long she's been pouting
  - Current emotional state
  - Scene context

---

## Architecture Overview

```
User Input
    │
    ▼
Wake Word Detection (InputManager)
    │
    ▼
WakeWordManager.get_active_wake_words(current_state)
    │
    ├─► Normal State: ["hey laura", "laura wake up"]
    └─► Pout State: ["quit being a baby", "i'm sorry", ...]
    │
    ▼
Context Injection (append to system instructions)
    │
    ▼
Movement Sequence Selection
    │
    ├─► "quit being a baby" → create_pout_exit_lunge_sequence()
    └─► "i'm sorry laura" → create_pout_exit_gentle_sequence()
    │
    ▼
MovementManager.queue_move(sequence)
    │
    ▼
100Hz Control Loop (non-interruptible sequence execution)
```

---

## Testing Plan

### Phase 1: Movement System
1. Test PoutPoseMove enters sleep pose correctly
2. Test discrete body rotations (0°, ±45°, ±90°)
3. Test antenna twitch patterns independently
4. Test breathing freeze during pout

### Phase 2: Sequences
1. Test lunge sequence timing and poses
2. Test gentle sequence timing and poses
3. Test face detection integration (user targeting)
4. Verify non-interruptible execution

### Phase 3: State Management
1. Test enter_pout_mode() from idle
2. Test exit_pout_mode() with different wake words
3. Test display state synchronization
4. Test face tracking enable/disable

### Phase 4: Wake Word Integration
1. Map Snowboy models to phrases
2. Test contextual wake word detection
3. Test context injection to LLM
4. Test scene_manager override logic

### Phase 5: End-to-End
1. Full flow: LLM triggers pout → user apologizes → gentle exit
2. Full flow: LLM triggers pout → user calls out → lunge exit
3. Test antenna animations during pout
4. Test body rotation during pout

---

## File Changes Summary

**New Files**:
- `pi_reachy_deployment/config/wake_words.yml` (90 lines)
- `pi_reachy_deployment/wake_word_manager.py` (250 lines)
- `pi_reachy_deployment/tools/pout_tools.py` (200 lines)
- `pi_reachy_deployment/POUT_MODE_IMPLEMENTATION_SUMMARY.md` (this file)

**Modified Files**:
- `pi_reachy_deployment/moves.py` (+500 lines)
- `pi_reachy_deployment/display/display_manager.py` (+4 lines, 3 changes)
- `pi_reachy_deployment/run_pi_reachy.py` (+150 lines initialization & handlers)

**Copied Assets**:
- `pout01.png`, `pout02.png`, `pout03.png` → LAURA images directory

**Total**: ~1200 new lines of code, 4 files modified, 4 files created

---

## Known Limitations

1. **Snowboy Model Mapping**: Need to create/train wake word models or establish naming convention
2. **Scene Manager Integration**: laura_agent.py not yet updated
3. **Wake Word Loop**: Context injection not yet integrated into LLM requests
4. **Audio Effects**: No muffled audio (user confirmed physical obstruction is sufficient)
5. **Additional Sequences**: Only 2 pout exit sequences implemented (lunge, gentle). Could add more:
   - `pout_peek` - Peek at user while staying in pout
   - `pout_turn_toward_user` - Rotate body to face user
   - `pout_antenna_emphasis` - Dramatic antenna swing

---

## Next Steps

**Priority 1 - Wake Word Integration**:
1. Decide on Snowboy model → phrase mapping strategy
2. Implement context injection into LLM conversation history
3. Add 'pout' to wake-responsive states
4. Test contextual wake word detection

**Priority 2 - Laura Agent**:
1. Update laura_agent.py system instructions
2. Add pout tool availability
3. Implement scene_manager decision logic
4. Test pout behavior in scene mode

**Priority 3 - Polish**:
1. Add additional movement sequences (peek, turn, emphasis)
2. Tune timing parameters (interpolation duration, hold times)
3. Add sound effects for pout enter/exit
4. Create comprehensive test suite

---

## Usage Example (Once Complete)

```python
# LLM decides Laura should pout
client.enter_pout_mode(initial_body_yaw_deg=-45)  # Turn away 45°

# While pouting, Laura can express frustration
client.pout_play_antenna_pattern("frustration_twitch")

# User: "quit being a baby laura"
# → Wake word detected
# → Context injected: "User called you out..."
# → Lunge sequence triggered
# → LLM responds with attitude

# OR user: "i'm sorry laura"
# → Gentle exit sequence
# → LLM can choose to forgive or stay stubborn
```

---

**Implementation Complete**: 92% (11/12 tasks)
**Ready for Testing**: Movement system, state handlers, display integration
**Remaining**: Wake word loop integration, laura_agent updates
