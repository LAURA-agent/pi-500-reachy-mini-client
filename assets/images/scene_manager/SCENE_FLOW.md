# Scene Manager Flow Architecture

## Overview
The Scene Manager orchestrates multi-agent conversations between gameboy devices, maintaining smooth playback through an N+2 generation pattern and centralized hub coordination.

## Key Components

### 1. iOS App (Control Interface)
- **Scene Setup View**
  - Select participating agents (Claude, Laura, Mari)
  - Set turn count (1-20)
  - Choose turn order mode:
    - `alternating`: Simple back-and-forth for 2 agents
    - `round_robin`: Cycles through all agents
    - `manual`: User specifies exact order
    - `director`: Claude decides who speaks when
  - Provide topic/context
- **Control Buttons**
  - Start: Initiate scene
  - Stop: Halt current scene
  - Continue: Trigger next segment with Director guidance

### 2. Scene Server (Port 8787)
- Receives scene configuration from iOS app
- Manages scene lifecycle
- Coordinates with Director for initial setup
- Handles continuation logic

### 3. Director Agent
- **Initial Setup Only**
  - Analyzes topic and participants
  - Creates scene objective
  - Determines first speaker
  - Sets overall tone
- **On Continue**
  - Provides new direction for next segment
  - Suggests plot twists or tone shifts
- **Stays Out During Scene**
  - No per-turn involvement
  - Agents operate autonomously

### 4. Turn Manager
- Executes the scene plan
- Manages speaker alternation
- Generates dialogue turns
- Distributes to hub

### 5. Mac Hub Bridge (Port 8766)
- Central relay point
- WebSocket connections to gameboy devices
- Routes dialogue to correct device
- Relays completion signals back
- Non-blocking message passing

### 6. Gameboy Receivers
- Display character visuals
- Generate TTS locally
- Play audio in sequence
- Send completion signals

## Critical Flow Sequence

### Bootstrap Phase (Turns 1-2)
```
1. Scene Request → Scene Server
2. Director creates scene plan
3. Generate Turn 1 → Send to Device 1
4. Turn 1 starts playing immediately
5. While Turn 1 plays: Generate Turn 2 → Queue on Device 2
6. Turn 1 completes → Signal sent
7. Turn 2 begins playing (no gap)
```

### N+2 Pattern (Turns 3+)
```
When Turn N completes:
  → Turn N+1 starts playing (already queued)
  → Turn N+2 begins generating

Example:
- Turn 1 completes → Turn 2 plays, Turn 3 generates
- Turn 2 completes → Turn 3 plays, Turn 4 generates
- Turn 3 completes → Turn 4 plays, Turn 5 generates
```

### Message Flow
```
Scene Manager → Hub → Device (dialogue distribution)
Device → Hub → Scene Manager (completion signals)
```

## Timing Constraints

### Bootstrap Requirements
- **Turn 1 + Turn 2 must be ready before playback starts**
- Turn 1 plays (~10 seconds of audio)
- Turn 2 must be generated and queued during this window
- No parallel generation (Turn 2 needs Turn 1's content for context)

### Generation Windows
- Each turn provides ~10 seconds of playback
- Next turn must generate within this window
- N+2 pattern provides buffer for slower generations

### Critical Flags
- `is_first`: Controls auto-playback behavior
  - `true`: Queues without playing (waits for signal)
  - `false`: Triggers continuation (plays when ready)
- Bootstrap turns (1 & 2) both marked `is_first: true`
- Subsequent turns marked `is_first: false`

## Configuration (scene_config.json)

### User-Facing Settings
```json
{
  "scene_defaults": {
    "turn_count": 6,
    "intensity": "medium",
    "save_transcript": true
  }
}
```

### Agent Registry
```json
{
  "available_agents": {
    "claude": { "device": "gameboy_1", "enabled": true },
    "laura": { "device": "gameboy_2", "enabled": true },
    "mari": { "device": "gameboy_3", "enabled": false }
  }
}
```

### Validation Rules
- No consecutive same speaker
- Minimum 2 agents
- Maximum 4 agents
- Turn count 1-20

## Failure Points & Solutions

### Problem: Gap Between Turns
**Cause**: Turn N+2 not ready when Turn N completes
**Solution**: Ensure bootstrap completes both turns before playback

### Problem: Out-of-Order Playback
**Cause**: Wrong `is_first` flags causing premature playback
**Solution**: Mark bootstrap turns correctly, validate queueing logic

### Problem: Blocking Webhooks
**Cause**: Scene Manager webhook handler blocking on generation
**Solution**: Use `asyncio.create_task()` for non-blocking generation

### Problem: Race Conditions
**Cause**: Turn 1 completion triggering generation before Turn 2 ready
**Solution**: Proper bootstrap sequencing, ignore early completion signals

## Scaling to 3+ Agents

### Bootstrap Changes
- Generate Turns 1, 2, 3 before playback
- N+3 pattern for 3 agents
- Round-robin or Director-controlled ordering

### Device Management
- Add `gameboy_3` for Mari
- Maintain WebSocket connections to all devices
- Ensure proper relay routing

### Future Enhancements
- Video generation during extended playback windows
- Dynamic agent joining mid-scene
- Parallel scene execution across multiple device groups

## Debug Commands

### Test Scene Start
```bash
curl -X POST http://localhost:8787/scene/start \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Who ate the cookies",
    "participants": ["claude", "laura"],
    "max_turns": 6
  }'
```

### Monitor Hub Connections
```bash
curl http://localhost:8766/connections
```

### Check Scene Status
```bash
curl http://localhost:8787/scene/status
```

## Key Principles

1. **Director Minimalism**: Setup and continue only, no per-turn involvement
2. **Buffer Maintenance**: Always have next turn ready before current completes
3. **Non-Blocking Operations**: All long-running tasks use async patterns
4. **Hub Centralization**: All device communication flows through hub
5. **Validation First**: Prevent invalid states rather than recover from them

## Common Mistakes to Avoid

- ❌ Parallel bootstrap generation (breaks context flow)
- ❌ Waiting for Director on every turn (adds latency)
- ❌ Blocking webhook handlers (causes timeouts)
- ❌ Ignoring completion signals (breaks N+2 pattern)
- ❌ Over-engineering configuration (users don't care about milliseconds)

## Success Metrics

- ✅ No gaps between character responses
- ✅ Natural conversation flow
- ✅ Sub-second response to control commands
- ✅ Clean scene termination
- ✅ Successful continuation with new direction