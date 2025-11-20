# Reachy Mini Gradio UI Implementation Plan

**Created:** November 19, 2025
**Status:** Planning Phase
**Purpose:** Comprehensive control interface for Reachy Mini robot with demo/presentation mode

---

## Current State (As of Nov 19, 2025)

### ✅ Completed: Platform Agnostic Refactor
- **Removed:** All vosk speech-to-text code (6 files deleted)
- **Cleaned:** client_config.py - removed all vosk model paths
- **Result:** App runs in Passive Mode on macOS with zero vosk references
- **API Server:** Running on `http://localhost:8888`
- **MCP Connection:** Successfully connects to `http://174.165.47.128:8765`

### Current Issues
- `StateTracker.STATE_CONFIGS` missing (interaction_loop.py:65)
- `AudioManager.start_input_stream()` method doesn't exist (handled with hasattr check)
- Interaction loop crashes when checking wake word detection config

### Architecture
```
Main App (passive mode)
├── No hardware input (input_manager = None)
├── No local transcription (speech_processor = None)
├── API Server on port 8888 (handles all input)
└── MCP Session to remote server (working)
```

---

## Wake Word → Action Mapping

Replace hardware wake words with Gradio UI buttons:

| Wake Word | Original Action | Gradio Button Function |
|-----------|----------------|------------------------|
| `Laura.pmdl` | Start conversation | **"Talk to Laura"** - Opens conversation flow |
| `GD_Laura.pmdl` | Enter pout mode | **"Pout Mode"** - Triggers frustrated state + audio |
| `Wake_up_Laura.pmdl` | Exit sleep | **"Wake Up"** - Exit sleep pose → idle |
| `claudecode.pmdl` | Code mode | **"Code Mode"** - Extended timeout for coding tasks |
| `sendnote.pmdl` | Transfer note | **"Send Note"** - Transfer note to Mac via API |
| `tookmycrazypills.pmdl` | Bluetooth mode | **"Bluetooth Mode"** - Enter bluetooth_ready state |

**Implementation Notes:**
- Each button calls corresponding command_handler method
- May need to expose new API endpoints for some actions
- Some actions already have API endpoints (see below)

---

## Existing API Endpoints (Port 8888)

Current endpoints available for Gradio integration:

### Pout Mode
- `POST /pout/trigger` - Enter pout mode
- `POST /pout/exit` - Exit pout mode
- `POST /pout/rotate` - Rotate during pout

### Display Control
- `POST /display/update` - Update display state/mood

### Speech/TTS
- `POST /speech/motion` - Trigger speech-synchronized antenna motion
- `POST /speech/motion/start` - Start speech motion with duration
- `POST /tts/conversation` - Full TTS conversation endpoint
- `POST /tts/working` - TTS working/thinking state

### Additional Endpoints Needed
- `POST /conversation/start` - Start MCP conversation flow
- `POST /speech/capture` - Start VAD-based speech capture
- `POST /mode/sleep` - Enter sleep mode
- `POST /mode/wake` - Exit sleep mode
- `POST /mode/code` - Enter code mode
- `POST /note/send` - Send note to Mac
- `GET /state/current` - Get current robot state
- `GET /camera/stream` - MJPEG camera stream (may already exist in daemon)

---

## Components to Integrate

### 1. Speech Capture System
**Current Issue:** No hardware microphone on Mac
**Solution Options:**
- Browser-based audio capture (Gradio Audio component)
- File upload for pre-recorded audio
- Text-only input for testing

**Features:**
- Start/Stop capture button
- Real-time waveform visualization (optional)
- VAD status indicator
- Transcription display

### 2. MCP Server Communication
**Current:** Already working via `http://174.165.47.128:8765`
**Session ID:** Managed by MCPSessionManager

**Gradio Integration:**
- Text input → send to MCP server
- Conversation history display
- Response streaming (if supported)
- Session status indicator

### 3. Scene Manager Integration
**Location:** `/Users/lauras/Desktop/laura/scene_manager/`
**Features to Integrate:**
- Start gameboy argument (Claude vs Laura debate)
- Monitor argument status
- Control argument parameters (topic, duration, etc.)
- Display argument transcript in real-time

**Key Files:**
- `gameboy_hub.py` - Main hub server
- `claude_gameboy.py` - Claude agent
- `orchestrator.py` - Scene orchestration

**API Endpoints Needed:**
- Start/stop argument
- Get argument status
- Stream argument transcript

### 4. Camera/Video Feed
**Sources:**
- Daemon MJPEG stream: `http://localhost:8100/api/camera/stream.mjpg`
- 3D simulator view (if available)

**Gradio Component:** `gr.Image()` with streaming updates

### 5. Move Controller (Port from HTML)
**Source:** `/Users/lauras/Desktop/laura/reachy_mini/move_controller.html`

**Features to Port:**
- Manual position control (6-DOF sliders)
- Pre-recorded move library (101 moves)
- Choreography builder
- Real-time pose visualization

**Implementation:** Convert HTML/JS to Gradio sliders + buttons

---

## Proposed Gradio UI Architecture

### Overall Layout: Tabbed Interface

```
reachy_mini_ui/
├── ui.py                    # Main Gradio interface
├── api_client.py           # API client for localhost:8888
├── mcp_client.py           # MCP server client wrapper
└── components/
    ├── conversation.py     # Chat interface
    ├── move_controller.py  # Robot control
    ├── camera_viewer.py    # Video feed
    └── scene_manager.py    # Gameboy integration
```

### Tab 1: Control Center (Primary Interface)

```
┌─────────────────────────────────────────────────────────────┐
│                     REACHY MINI CONTROL                      │
├──────────────────────────┬──────────────────────────────────┤
│                          │  STATE: Idle                      │
│   CAMERA/SIMULATOR       │  MOOD: Casual                     │
│   [Live Video Feed]      │  SESSION: Connected               │
│                          │                                   │
│   640x480                ├───────────────────────────────────┤
│                          │  QUICK ACTIONS                    │
│                          │  [Talk to Laura] [Pout Mode]      │
│                          │  [Wake Up] [Code Mode]            │
│                          │  [Send Note] [Bluetooth]          │
├──────────────────────────┴───────────────────────────────────┤
│  CONVERSATION                                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Chat History (scrollable)                              │ │
│  │ User: Hello Laura                                      │ │
│  │ Laura: Hi! How can I help?                             │ │
│  └────────────────────────────────────────────────────────┘ │
│  [Text Input__________________________________] [Send]       │
│  [Audio Input (optional)]                                   │
│  [🎤 Start Speech Capture]                                  │
└─────────────────────────────────────────────────────────────┘
```

### Tab 2: Move Controller

```
┌─────────────────────────────────────────────────────────────┐
│                    MANUAL CONTROL                            │
├──────────────────────────┬──────────────────────────────────┤
│  POSITION CONTROL        │  ROTATION CONTROL                 │
│  X: [slider] [-3 to 3]   │  Yaw:   [slider] [-180 to 180]   │
│  Y: [slider] [-3 to 3]   │  Pitch: [slider] [-90 to 90]     │
│  Z: [slider] [-3 to 3]   │  Roll:  [slider] [-180 to 180]   │
│                          │                                   │
│  ANTENNA CONTROL         │  [Apply Position]                 │
│  Left:  [slider]         │  [Reset to Neutral]               │
│  Right: [slider]         │                                   │
│  [Bind] [Inverse]        │                                   │
├──────────────────────────┴───────────────────────────────────┤
│  PRE-RECORDED MOVES                                          │
│  Dances: [side_sway] [jackson] [dizzy_spin] ...             │
│  Emotions: [amazed] [confused] [happy] ...                   │
│  [Execute Selected] [Stop]                                   │
├─────────────────────────────────────────────────────────────┤
│  CHOREOGRAPHY BUILDER                                        │
│  Routine: [Move 1] [Move 2] [Move 3] ...                    │
│  BPM: [120] Amplitude: [1.0] Cycles: [1]                    │
│  [Add Move] [Remove] [Export JSON]                           │
└─────────────────────────────────────────────────────────────┘
```

### Tab 3: Scene Manager

```
┌─────────────────────────────────────────────────────────────┐
│                  GAMEBOY ARGUMENT SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│  ARGUMENT CONFIGURATION                                      │
│  Topic: [Text Input_________________________]                │
│  Duration: [5] minutes   Intensity: [Medium ▼]              │
│  Claude Personality: [Logical ▼]  Laura: [Emotional ▼]      │
│                                                              │
│  [🎭 Start Argument] [⏸ Pause] [⏹ Stop]                      │
├─────────────────────────────────────────────────────────────┤
│  LIVE TRANSCRIPT                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Claude: I think we should approach this logically...   │ │
│  │ Laura: But what about the human element?               │ │
│  │ Claude: Statistics show...                              │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  STATUS: Running  |  Exchanges: 12  |  Time: 02:34          │
└─────────────────────────────────────────────────────────────┘
```

### Tab 4: Configuration

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM CONFIGURATION                      │
├──────────────────────────┬──────────────────────────────────┤
│  TTS SETTINGS            │  NETWORK SETTINGS                 │
│  Provider: [ElevenLabs▼] │  MCP Server: [174.165.47.128:8765]│
│  Voice: [Laura ▼]        │  API Server: [localhost:8888]     │
│  Model: [flash_v2_5 ▼]   │  Daemon: [localhost:8100]         │
│  [Test Voice]            │  [Test Connection]                │
├──────────────────────────┼──────────────────────────────────┤
│  BEHAVIOR SETTINGS       │  DISPLAY SETTINGS                 │
│  Sleep Timeout: [5] min  │  Window Size: [640x480 ▼]         │
│  Auto Wake: [✓]          │  Profile: [Normal ▼]              │
│  Face Tracking: [✓]      │  [Show/Hide Display Window]       │
│  Breathing: [✓]          │                                   │
├──────────────────────────┴───────────────────────────────────┤
│  [Save Configuration] [Reset to Defaults]                    │
└─────────────────────────────────────────────────────────────┘
```

### Tab 5: Extensions (Future)

```
┌─────────────────────────────────────────────────────────────┐
│                    THIRD-PARTY INTEGRATIONS                  │
├─────────────────────────────────────────────────────────────┤
│  AVAILABLE PLUGINS                                           │
│  [ ] Stable Diffusion Image Generation                       │
│  [ ] Custom Choreography Loader                              │
│  [ ] External Camera Feeds                                   │
│  [ ] Voice Cloning Tools                                     │
│                                                              │
│  [Load Plugin] [Configure] [Remove]                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)
1. Create `ui/gradio_app.py` with basic tab structure
2. Integrate into main app startup sequence
3. Create API client wrapper for localhost:8888
4. Implement state display (read-only)

### Phase 2: Control Center (Week 2)
1. Text conversation interface
2. Quick action buttons (pout, wake, etc.)
3. Camera feed streaming
4. Basic speech capture (file upload initially)

### Phase 3: Move Controller (Week 3)
1. Manual position sliders
2. Pre-recorded move buttons
3. Choreography builder (simplified)

### Phase 4: Scene Manager (Week 4)
1. Argument start/stop controls
2. Transcript display
3. Status monitoring

### Phase 5: Configuration & Polish (Week 5)
1. Settings persistence
2. Error handling
3. UI refinements
4. Documentation

---

## Technical Implementation Details

### Gradio Server Integration

**Option 1: Separate Process (Recommended for development)**
```python
# ui/launch_gradio.py
import gradio as gr
from gradio_app import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

**Option 2: Integrated (Production)**
```python
# core/app_orchestrator.py
async def start_gradio_server(self):
    from ui.gradio_app import create_ui
    self.gradio_demo = create_ui(self)
    threading.Thread(
        target=lambda: self.gradio_demo.launch(
            server_port=7860,
            prevent_thread_lock=True
        ),
        daemon=True
    ).start()
```

### API Client Pattern

```python
# ui/api_client.py
import httpx

class ReachyAPIClient:
    def __init__(self, base_url="http://localhost:8888"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def trigger_pout(self):
        response = await self.client.post(f"{self.base_url}/pout/trigger")
        return response.json()

    async def start_conversation(self, text: str):
        response = await self.client.post(
            f"{self.base_url}/conversation/start",
            json={"text": text}
        )
        return response.json()
```

### State Updates (Real-time)

Use Gradio's `gr.State()` + periodic updates:

```python
def create_ui():
    with gr.Blocks() as demo:
        state_display = gr.Textbox(label="Robot State", interactive=False)

        # Update every 1 second
        demo.load(
            fn=update_state,
            inputs=None,
            outputs=state_display,
            every=1
        )

    return demo

async def update_state():
    client = ReachyAPIClient()
    state = await client.get_current_state()
    return state["state"]
```

---

## Missing Features / Questions

### To Decide:
1. **Audio Input:** Browser-based or file upload for speech capture?
2. **Camera Stream:** Direct MJPEG or periodic image updates?
3. **Real-time Updates:** Polling interval vs WebSocket?
4. **Session Persistence:** Save conversation history where?
5. **Multi-user:** Support multiple simultaneous Gradio sessions?

### To Implement:
- **Notification System** - How to display incoming notifications in Gradio?
- **Error Recovery** - UI behavior when API calls fail
- **Authentication** - Password protect Gradio interface?
- **Mobile Layout** - Responsive design for tablets?
- **Keyboard Shortcuts** - Hotkeys for common actions?

---

## File Structure

```
pi-500-reachy-mini-client-1/
├── ui/
│   ├── __init__.py
│   ├── gradio_app.py           # Main UI creation
│   ├── api_client.py           # Localhost API wrapper
│   ├── mcp_client.py           # MCP server wrapper
│   ├── components/
│   │   ├── conversation.py     # Chat interface
│   │   ├── move_controller.py  # Robot control
│   │   ├── camera_viewer.py    # Video feed
│   │   ├── scene_manager.py    # Gameboy integration
│   │   └── config_panel.py     # Settings
│   └── launch_gradio.py        # Standalone launcher
├── core/
│   └── app_orchestrator.py     # (Modified to start Gradio)
└── GRADIO_UI_PLAN.md           # This document
```

---

## Next Steps

1. **Fix Current Errors:**
   - Remove StateTracker.STATE_CONFIGS dependency
   - Fix interaction_loop for passive mode
   - Test app stability without wake word detection

2. **Create API Endpoints:**
   - Add missing endpoints for Gradio actions
   - Document API with OpenAPI/Swagger

3. **Start Phase 1:**
   - Create basic Gradio UI structure
   - Implement tab navigation
   - Connect to existing API endpoints

4. **Testing:**
   - Test on Mac (development)
   - Test on Pi (production with hardware)
   - Verify all buttons trigger correct actions

---

## References

- **Move Controller HTML:** `/Users/lauras/Desktop/laura/reachy_mini/move_controller.html`
- **Stable Diffusion UI:** `/Users/lauras/Desktop/laura/reachy_mini/UI-resources/stable-diffusion-webui/`
- **Existing Gradio Apps:** `/Users/lauras/Desktop/laura/archival files/gradio_app.py`
- **Scene Manager:** `/Users/lauras/Desktop/laura/scene_manager/`
- **API Endpoints:** `communication/api_server.py`

---

**End of Planning Document**
