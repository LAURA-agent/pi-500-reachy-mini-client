# Reachy Mini Gradio UI

Comprehensive control interface for Reachy Mini with 8 Chrome-style tabs.

## Features

### 8-Tab Interface

1. **💬 Conversation** - Main chat interface with text/audio input, chat history
2. **🎮 Manual Control** - 6-DOF sliders for head position and antenna control
3. **🎵 Choreography** - Pre-recorded move library and routine builder (coming soon)
4. **📹 Robot Viewer** - 3D simulator feed and live pose monitoring (coming soon)
5. **🎤 STT Config** - Speech-to-text provider selection (ElevenLabs Scribe / Whisper HF)
6. **🔊 TTS Config** - Text-to-speech provider selection (ElevenLabs / Local)
7. **🧠 Inference** - Backend selection (MCP Server / Messages API)
8. **⚙️ System** - Movement, breathing, and network configuration

## Installation

```bash
# Install Gradio UI dependencies
pip install -r requirements_gradio_ui.txt
```

## Usage

### 1. Start the Main Robot App (API Server on port 8888)

```bash
python main.py
```

This starts the robot control system with API endpoints on http://localhost:8888

### 2. Launch the Gradio UI (separate process)

```bash
python launch_gradio_ui.py
```

The UI will be available at http://localhost:7860

## Architecture

```
┌─────────────────────────────────────────────┐
│         Gradio UI (Port 7860)                │
│  ┌────────────────────────────────────────┐ │
│  │  8 Tabs: Conversation, Control, Config  │ │
│  └─────────────────┬────────────────────────┘ │
└────────────────────┼──────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────┐
│      Main Robot App (Port 8888)              │
│  ┌────────────────────────────────────────┐ │
│  │  API Server (aiohttp)                   │ │
│  │  - Config endpoints                     │ │
│  │  - State endpoints                      │ │
│  │  - Conversation endpoints               │ │
│  │  - Pout/Display/TTS endpoints           │ │
│  └─────────────────┬────────────────────────┘ │
│                    │                          │
│  ┌─────────────────┴────────────────────────┐ │
│  │  RobotController, StateTracker, etc.     │ │
│  └──────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
          Reachy Mini Daemon (Port 8100)
```

## Configuration System

The UI manages 4 separate config files:

- **`config_files/stt_config.json`** - Speech-to-text settings
- **`config_files/tts_config.json`** - Text-to-speech settings
- **`config_files/inference_config.json`** - MCP vs Messages API backend
- **`config_files/movement_config.json`** - Breathing, face tracking, speech motion

Each config tab allows:
- Viewing current settings
- Editing JSON configuration
- Saving changes to file
- Reloading from disk

Changes are saved via API endpoints and immediately applied to the running system.

## API Endpoints Used

### Config Management
- `GET /config/{config_name}` - Get current config
- `POST /config/{config_name}` - Update config and save to file
- `POST /config/reload/{config_name}` - Reload config from disk
- `POST /config/reload_all` - Reload all configs

### Robot State
- `GET /state/current` - Get current robot state (idle, speaking, etc.)

### Conversation
- `POST /conversation/start` - Start conversation with text input

### Existing Endpoints
- `POST /pout/trigger` - Enter pout mode
- `POST /pout/exit` - Exit pout mode
- `POST /display/update` - Update display state/mood
- `POST /speech/motion` - Trigger speech-synchronized motion
- `POST /tts/conversation` - Full TTS conversation

## Styling

The UI uses Chrome-style tabs with the **Pollen Robotics color palette**:

- **Navy Blue** (#2B4C7E) - Primary brand color, tab bar
- **Light Blue** (#3bb0d1) - Active elements, borders
- **Light Green** (#3dde99) - Success states
- **Yellow Gold** (#ffc261) - Accents
- **Pink** (#ff6170) - Danger zones, errors
- **White** (#FFFFFF) - Content background

Custom CSS is defined in `ui/gradio_app.py` (CHROME_TAB_CSS constant).

## Development Status

### ✅ Implemented
- Main Gradio app with 8 tabs
- Chrome-style tab CSS
- Conversation tab (text input, chat history)
- Manual control tab (6-DOF sliders)
- All 4 config tabs (STT, TTS, Inference, System)
- Config management API endpoints
- Modular config file system

### 🚧 In Progress
- Audio input integration with STT providers
- Conversation flow (MCP vs Messages API)
- Manual control API integration

### 📋 Planned
- Choreography builder (move library, BPM, export)
- Robot viewer (MJPEG stream, pose charts)
- Quick action buttons (wake, sleep, pout, stop)
- Real-time state monitoring
- Test buttons for STT/TTS

## Troubleshooting

**Gradio UI won't start:**
- Make sure gradio is installed: `pip install gradio`
- Check Python version (3.10+)

**Can't connect to API:**
- Verify main app is running: `curl http://localhost:8888/state/current`
- Check logs for errors

**Config changes not saved:**
- Ensure `config_files/` directory exists
- Check file permissions

## Next Steps

1. **Complete conversation flow** - Integrate STT → Inference → TTS pipeline
2. **Add choreography builder** - Port from move_controller.html
3. **Implement robot viewer** - MJPEG stream + WebSocket pose data
4. **Test end-to-end** - Full workflow from speech input to robot movement

---

**Created:** November 19, 2025
**Status:** MVP functional, actively developing
