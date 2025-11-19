# Pi Reachy Deployment - Claude Code Instructions

## Repository Purpose

This repository manages a Reachy Mini robot deployment on a Raspberry Pi. The main script (`run_pi_reachy.py`) runs the robot's main event loop once the reachy-mini daemon is operational.

**Core Philosophy:**
- This script will be continuously built upon as we add/remove functionality
- New features should be added as **new states** with entry/exit sequences
- **DO NOT modify existing state behavior** - add new states instead
- The robot operates primarily in `idle` or `sleep` state

---

## Architecture: Robot State Manager

The system uses a state-based architecture where the robot transitions between distinct behavioral states:

### Current States

**Core States:**
- **`idle`** - Normal conversational state, breathing motion active, face tracking enabled
- **`sleep`** - Low-power state after 5min timeout, hunched sleep pose, breathing paused, wake word detection active
- **`listening`** - Actively capturing user speech, breathing paused
- **`thinking`** - Processing user input with MCP server
- **`speaking`** - Playing TTS response with speech-synchronized antenna motion
- **`pout`** - Upset/frustrated state, hunched pose, can still do full TTS conversations
- **`bluetooth_ready`** - Monitoring Bluetooth audio stream for sentence detection
- **`bluetooth_playing`** - Playing analyzed Bluetooth audio with synchronized motion
- **`code`** - Code execution state (placeholder)
- **`execution`** - Execution state (placeholder)
- **`error`** - Error recovery (minimal - just returns to idle)

### State Transitions

States are managed via `StateTracker` (simple) or `RobotStateMachine` (formal state machine with enforced transitions).

**State Change Callback:** `_on_state_change()` triggers physical behaviors:
- Face tracking enable/disable
- Sleep pose entry/exit
- Breathing pause/resume

---

## Adding New Features: The State Pattern

### When to Add a New State

Add a new state when:
- The feature requires distinct physical behavior (pose, breathing, movement pattern)
- The feature has clear entry and exit conditions
- The feature should be mutually exclusive with normal conversation

### How to Add a New State

1. **Define the state** in `StateTracker.VALID_STATES` or `RobotStateMachine`
2. **Create entry method** - `enter_<state>_mode()` that sets up the physical behavior
3. **Create exit method** - `exit_<state>_mode()` that cleans up and returns to neutral
4. **Add to `_on_state_change()`** callback if needed for automatic transitions
5. **Handle in main loop** - add conditional logic for state-specific wake events

**Example Pattern:**
```python
def enter_choreography_mode(self):
    """Enter choreography state - run dance sequence."""
    print("[CHOREOGRAPHY] Starting sequence...")
    self.state_tracker.update_state("choreography")
    # Pause breathing, disable face tracking
    self.movement_manager.pause_breathing()
    self.camera_worker.set_head_tracking_enabled(False)
    # Trigger choreography scripts (desktop_viewer.py, react_agent.py)
    # ...

def exit_choreography_mode(self):
    """Exit choreography and return to idle."""
    print("[CHOREOGRAPHY] Sequence complete, returning to idle...")
    self.movement_manager.resume_breathing()
    self.camera_worker.set_head_tracking_enabled(True)
    self.state_tracker.update_state("idle")
```

---

## Planned State Extensions

### Choreography State (Not Yet Implemented)

**Purpose:** Execute multi-process choreographed dance sequences

**Scope:**
- Triggers sequences in `choreography/desktop_viewer.py`
- Coordinates with `choreography/react_agent.py`
- May involve main script coordination
- Requires inter-process communication

**Entry:** Wake word or API trigger from idle/sleep
**Exit:** Sequence completion → return to idle

### Manual Movement State (Future)

**Purpose:** Direct manual control for testing/adjustment

**Entry:** API or command trigger
**Exit:** Manual dismiss or timeout

---

## Key Constraints

### Movement System
- **Breathing and face tracking are mutually exclusive with many states** - pause breathing when doing discrete movements
- **Speech offsets are ONLY for speaking state** - antenna oscillation synchronized with TTS phonemes
- **Pout and sleep use the same physical pose** (SLEEP_HEAD_POSE) but different behaviors

### State Behavior Rules
- **Moods are ONLY for pygame display during speaking state** - not for movement control
- **Sleep state uses pre-recorded audio OR full TTS** (both work, no restriction)
- **Pout state allows full TTS conversation** - just in hunched pose
- **Face tracking ONLY in idle state** - disabled everywhere else

### TTS and Audio
- **Always include TTS markers** in responses: `<!-- TTS: "text" -->`
- **Speech motion synchronization** happens via claude-to-speech plugin stop hook
- **No mood-based movements** - cc_plugin_mood state was removed

---

## Common Pitfalls

❌ **DON'T modify existing state behavior** - add new states instead
❌ **DON'T add moods to idle/sleep states** - moods are display-only for speaking
❌ **DON'T forget to pause breathing** when entering states with discrete movements
❌ **DON'T assume face tracking works in non-idle states** - it's disabled
❌ **DON'T use blocking operations in main loop** - use async/await or background tasks

✅ **DO create entry/exit methods** for new states
✅ **DO clean up properly** - resume breathing, re-enable face tracking on exit
✅ **DO use state callbacks** for automatic physical behavior triggers
✅ **DO check existing state implementations** (pout, sleep) as examples
✅ **DO include TTS markers** in every response

---

## Development Notes

- Main loop runs at ~20Hz (0.05s sleep)
- Wake word detection uses Snowboy with .pmdl models
- MCP server handles conversation via LAURA agent
- Daemon (port 8100) handles low-level motor control
- TTS server (port 5001) generates speech audio
- HTTP endpoints (port 8888) for external control

See `CLAUDE.md` in main `/home/user/reachy` folder for higher-level system documentation.
