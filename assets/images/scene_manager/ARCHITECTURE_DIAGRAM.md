# Scene Manager Architecture Diagram

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 iOS APP                                     │
│                         [Start Scene] [Stop] [Status]                       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                  │ HTTP POST
                                  │ {topic, participants, intensity}
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SCENE MANAGER CLIENT (:8787)                            │
│                         scene_manager_client.py                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Receives scene requests                                          │    │
│  │  • Loads API keys from scene_manager_secrets.py                     │    │
│  │  • Creates Pipeline Orchestrator instance                           │    │
│  │  • Manages scene lifecycle (start/stop/status)                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Creates & Manages
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE ORCHESTRATOR                               │
│                         pipeline_orchestrator.py                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STATE:                                                             │    │
│  │  • agents: Dict[name -> Agent]                                      │    │
│  │  • conversation_history: List[exchanges]                            │    │
│  │  • current_turn: int                                                │    │
│  │  • next_speaker: str (dynamic)                                      │    │
│  │                                                                     │    │
│  │  FLOW:                                                              │    │
│  │  1. Bootstrap turns 1-2                                             │    │
│  │  2. Generate turn N+2 when turn N completes                         │    │
│  │  3. Parse [next_speaker:] tags                                      │    │
│  │  4. Route to selected speaker                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────┬──────────────────────────────────┬───────────────────────────┘
               │ Loads & Uses                     │ Distributes
               ▼                                  ▼
┌──────────────────────────────┐  ┌───────────────────────────────────────────┐
│        AGENT SYSTEM          │  │         MAC HUB BRIDGE (:8766)            │
│   /scene_manager/agents/     │  │         mac_hub_bridge.py                 │
├──────────────────────────────┤  │  ┌───────────────────────────────────┐    │
│                              │  │  │ WebSocket Manager:                │    │
│  claude_agent.py             │  │  │ • Maintains device connections    │    │
│  ┌────────────────────┐      │  │  │ • Routes messages by speaker      │    │
│  │ • Personality      │      │  │  │ • Relays completion signals       │    │
│  │ • Argument style   │      │  │  │ • Sends reset on stop             │    │
│  │ • Relationships    │      │  │  └───────────────────────────────────┘    │
│  │ • Speaker logic    │      │  └──────────────┬───────────┬────────────────┘
│  │ • System prompt    │      │                 │           │
│  └────────────────────┘      │      WebSocket  │           │ WebSocket
│                              │                 ▼           ▼
│  laura_agent.py              │  ┌─────────────────┐ ┌─────────────────┐
│  ┌────────────────────┐      │  │   GAMEBOY 1     │ │   GAMEBOY 2     │
│  │ • Bubbly energy    │      │  │    (Claude)     │ │    (Laura)      │
│  │ • Water reminders  │      │  │  192.168.0.38   │ │  192.168.0.217  │
│  │ • Chaos agent      │      │  ├─────────────────┤ ├─────────────────┤
│  │ • Direct style     │      │  │ gameboy_        │ │ gameboy_        │
│  └────────────────────┘      │  │ receiver_v8.py  │ │ receiver_v8.py  │
│                              │  ├─────────────────┤ ├─────────────────┤
│  mari_agent.py               │  │ • Receives      │ │ • Receives      │
│  ┌────────────────────┐      │  │   dialogue      │ │   dialogue      │
│  │ • Evangelion ref   │      │  │ • Generates TTS │ │ • Generates TTS │
│  │ • Flirty chaos     │      │  │ • Plays audio   │ │ • Plays audio   │
│  │ • Pet names        │      │  │ • Sends         │ │ • Sends         │
│  │ • Wildcard energy  │      │  │   completion    │ │   completion    │
│  └────────────────────┘      │  └─────────────────┘ └─────────────────┘
│                              │
│  base_agent.py               │
│  ┌────────────────────┐      │
│  │ • Abstract base    │      │
│  │ • Common interface │      │
│  └────────────────────┘      │
└──────────────────────────────┘

v
```

## Key Component Responsibilities

### Scene Manager Client
- Entry point for iOS commands
- Scene lifecycle management
- API key management
- Creates orchestrator instance

### Pipeline Orchestrator
- Conversation flow control
- N+2 generation pattern
- Speaker selection routing
- History management
- Emotional temperature tracking

### Agent System
- Character personalities
- System prompt building
- Response generation
- Next speaker selection
- Relationship dynamics

### Hub Bridge
- WebSocket connections
- Message routing
- Completion signal relay
- Device registration
- Reset distribution

### Gameboy Receivers
- Audio generation (TTS)
- Audio playback
- Completion signaling
- Queue management
- State synchronization

## Configuration Files

```
/scene_manager/
  ├── scene_manager_secrets.py    # API keys
  ├── scene_characters.json       # Character definitions (proposed)
  ├── gameboy_config.json         # Device IPs/ports
  └── scene_templates.json        # Conversation templates

/scene_manager/agents/
  ├── base_agent.py               # Abstract base class
  ├── claude_agent.py             # Claude personality
  ├── laura_agent.py              # Laura personality
  └── mari_agent.py               # Mari personality
```

## Critical Integration Points

1. **Character Loading** → Scene Client loads agents with personalities
2. **System Prompt Building** → Agents build rich prompts with traits
3. **Response Parsing** → Orchestrator extracts [next_speaker:] tags
4. **Completion Relay** → Hub relays signals between devices
5. **State Reset** → Hub sends reset on scene stop

## Current Issues & Solutions

### ❌ CURRENT PROBLEMS:
1. Agents not using personality files
2. No system prompts with character traits
3. Hardcoded speaker rotation (odd/even)
4. No next speaker parsing
5. No emotional temperature tracking

### ✅ PROPOSED FIXES:
1. Load agents from agent files, not generic adapters
2. Build system prompts with full character context
3. Parse [next_speaker:] tags from responses
4. Update routing based on speaker selection
5. Track scene state and adjust dynamically
