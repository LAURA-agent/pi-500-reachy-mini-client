#!/usr/bin/env python3
"""
Standalone Gameboy WebSocket Hub
Minimal communication bridge for scene manager demo
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gameboy Hub", version="1.0.0")

# WebSocket connections: speaker -> websocket
connections: Dict[str, WebSocket] = {}

# IP to speaker mapping
ip_to_speaker = {
    "192.168.0.38": "claude",
    "192.168.0.215": "laura",  # Updated from .217
    "172.20.10.2": "claude",
    "172.20.10.3": "laura"
}

# Track active scene to route playback_complete messages
active_scene_callback = None

# Store tools with timing info, keyed by turn number
# Structure: {turn: {"tool": tool_data, "timing": "start"/"end"}}
pending_tools: Dict[int, Dict[str, Any]] = {}


class DialoguePayload(BaseModel):
    """Dialogue distribution request from turn_manager"""
    speaker: str
    turn: int
    text: str
    is_first: bool  # True for turn 1, False for all others
    mood: str = None
    speaking_image: str = None
    idle_image: str = None
    tool: Dict = None  # Laura's tool data


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket connection for gameboy devices
    Waits for device_registration message to identify device

    Devices send:
    - {"type": "device_registration", "speaker": "claude"/"laura"}
    - {"type": "playback_complete", "turn": N}

    Devices receive:
    - {"type": "dialogue", "speaker": "laura", "text": "...", "is_first": true/false, ...}
    - {"type": "play_signal"}  # Sent 0.2s after dialogue distribution for non-first turns
    """
    await websocket.accept()
    speaker = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            message_type = message.get("type")

            if message_type == "device_registration":
                speaker = message.get("speaker")
                if speaker:
                    connections[speaker] = websocket
                    logger.info(f"📱 Device registered: {speaker}")
                else:
                    logger.error("❌ Registration missing speaker")

            elif message_type == "playback_complete":
                turn = message.get("turn")
                logger.info(f"🔔 Playback complete from {speaker}: Turn {turn}")

                # Execute any pending "end" timing tools for this turn (only if from Laura)
                if speaker == "laura" and turn in pending_tools:
                    tool_data = pending_tools.pop(turn)
                    if tool_data["timing"] == "end":
                        logger.info(f"🔧 Executing end-timing tool for turn {turn}: {tool_data['tool'].get('action')}")
                        await execute_tool(tool_data["tool"])

                # Forward to scene_server webhook
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(
                            "http://localhost:9000/playback_complete",
                            json={"turn": turn, "from_device": speaker}
                        )
                        logger.info(f"✅ Forwarded playback_complete to scene_server")
                    except Exception as e:
                        logger.error(f"❌ Failed to forward playback_complete: {e}")

    except WebSocketDisconnect:
        logger.info(f"📱 Device disconnected: {speaker}")
        if speaker and speaker in connections:
            del connections[speaker]


@app.post("/api/dialogue/distribute_single")
async def distribute_dialogue(payload: DialoguePayload):
    """
    Receive dialogue from turn_manager and send to appropriate device

    For is_first=True: Send immediately and device plays
    For is_first=False: Send immediately, then send play_signal 0.2s later

    Tool timing synchronization:
    - "start" timing: Execute when play_signal is sent to Laura (as she begins speaking)
    - "end" timing: Execute when Laura's playback_complete arrives (after she finishes)
    """
    speaker = payload.speaker

    if speaker not in connections:
        logger.error(f"❌ Device {speaker} not connected")
        return {"status": "error", "message": f"Device {speaker} not connected"}

    ws = connections[speaker]

    # Build message for device
    message = {
        "type": "dialogue",
        "speaker": payload.speaker,
        "turn": payload.turn,
        "text": payload.text,
        "is_first": payload.is_first
    }

    # Add visual data if present
    if payload.mood:
        message["mood"] = payload.mood
    if payload.speaking_image:
        message["speaking_image"] = payload.speaking_image
    if payload.idle_image:
        message["idle_image"] = payload.idle_image

    # Send dialogue to device
    await ws.send_text(json.dumps(message))
    logger.info(f"📤 Sent turn {payload.turn} to {speaker} (is_first={payload.is_first})")

    # Handle tool if present (only Laura has tools)
    if payload.tool and speaker == "laura" and payload.tool.get("action"):
        tool = payload.tool
        timing = tool.get("timing", "end")

        # Store tool with timing info for execution at correct moment
        pending_tools[payload.turn] = {
            "tool": tool,
            "timing": timing
        }
        logger.info(f"🔧 Stored {timing}-timing tool for turn {payload.turn}: {tool.get('action')}")

    # For non-first turns, send play signal after 0.2s delay
    if not payload.is_first:
        async def send_play_signal():
            await asyncio.sleep(0.1)
            if speaker in connections:  # Check still connected
                await connections[speaker].send_text(json.dumps({"type": "play", "turn": payload.turn}))
                logger.info(f"▶️  Sent play_signal to {speaker} for turn {payload.turn}")

                # If this is Laura and she has a "start" timing tool, execute it now
                if speaker == "laura" and payload.turn in pending_tools:
                    tool_data = pending_tools[payload.turn]
                    if tool_data["timing"] == "start":
                        logger.info(f"🔧 Executing start-timing tool for turn {payload.turn}: {tool_data['tool'].get('action')}")
                        await execute_tool(tool_data["tool"])
                        # Remove from pending since it's executed
                        pending_tools.pop(payload.turn)

        asyncio.create_task(send_play_signal())

    return {"status": "ok", "device": speaker, "turn": payload.turn}


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Gameboy Hub",
        "port": 1025,
        "connected_devices": list(connections.keys())
    }


@app.get("/api/status")
async def api_status():
    """Status endpoint for iOS app compatibility"""
    return {
        "status": "connected",
        "connected_devices": list(connections.keys())
    }


@app.get("/scene/status")
async def scene_status():
    """Scene status - proxies to scene_server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:9000/scene/status")
            return response.json()
        except Exception as e:
            logger.error(f"Error getting scene status: {e}")
            return {"state": "error", "message": str(e)}


@app.post("/scene/start")
async def scene_start(request: dict):
    """Start scene - proxies to scene_server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("http://localhost:9000/scene/start", json=request)
            return response.json()
        except Exception as e:
            logger.error(f"Error starting scene: {e}")
            return {"status": "error", "message": str(e)}


@app.post("/scene/stop")
async def scene_stop():
    """Stop scene - proxies to scene_server and resets gameboys"""
    # Stop the scene on scene_server
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("http://localhost:9000/scene/stop")
            result = response.json()
        except Exception as e:
            logger.error(f"Error stopping scene: {e}")
            return {"status": "error", "message": str(e)}

    # Clear pending tools
    pending_tools.clear()
    logger.info("🔧 Cleared pending tools")

    # Send conversation_reset to all connected gameboys
    reset_message = json.dumps({"type": "conversation_reset"})
    for speaker, websocket in connections.items():
        try:
            await websocket.send_text(reset_message)
            logger.info(f"🔄 Sent conversation_reset to {speaker}")
        except Exception as e:
            logger.error(f"❌ Failed to send reset to {speaker}: {e}")

    return result


@app.post("/scene/continue")
async def scene_continue(request: dict):
    """Continue scene - proxies to scene_server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("http://localhost:9000/scene/continue", json=request)
            return response.json()
        except Exception as e:
            logger.error(f"Error continuing scene: {e}")
            return {"status": "error", "message": str(e)}


@app.post("/api/manipulate_star")
async def manipulate_star(request: dict):
    """
    Manipulate Claude's bouncing star
    Sends command to Claude's gameboy device
    """
    action = request.get("action")
    parameters = request.get("parameters", {})

    if "claude" not in connections:
        logger.error("❌ Claude device not connected")
        return {"status": "error", "message": "Claude device not connected"}

    ws = connections["claude"]

    # Send star manipulation command
    message = {
        "type": "star_manipulation",
        "action": action,
        "parameters": parameters
    }

    await ws.send_text(json.dumps(message))
    logger.info(f"⭐ Sent star manipulation to Claude: {action}")

    return {"status": "ok", "action": action}


@app.post("/api/update_auxiliary_display")
async def update_auxiliary_display(request: dict):
    """
    Update auxiliary display device
    Sends command to auxiliary_display device
    """
    image = request.get("image")

    if not image:
        return {"status": "error", "message": "Missing image parameter"}

    if "auxiliary_display" not in connections:
        logger.warning("⚠️ Auxiliary display not connected")
        return {"status": "warning", "message": "Auxiliary display not connected"}

    ws = connections["auxiliary_display"]

    # Send display update command
    message = {
        "type": "display_image",
        "image": image
    }

    await ws.send_text(json.dumps(message))
    logger.info(f"🖥️ Sent display update to auxiliary: {image}")

    return {"status": "ok", "image": image}


@app.post("/api/queue_tool")
async def queue_tool(request: dict):
    """
    Queue a tool for execution at a specific turn
    Called by Laura's agent during tool use loop
    """
    turn = request.get("turn")
    action = request.get("action")
    timing = request.get("timing", "end")
    parameters = request.get("parameters", {})

    if not turn or not action:
        return {"status": "error", "message": "Missing turn or action"}

    # Store tool in pending_tools for execution at correct moment
    pending_tools[turn] = {
        "tool": {
            "action": action,
            "parameters": parameters
        },
        "timing": timing
    }

    logger.info(f"🔧 Queued tool for turn {turn}: {action} with '{timing}' timing")
    return {
        "status": "ok",
        "turn": turn,
        "action": action,
        "timing": timing,
        "message": f"Tool '{action}' queued for turn {turn} with '{timing}' timing"
    }


@app.post("/api/queue_auxiliary_display")
async def queue_auxiliary_display(request: dict):
    """
    Queue an auxiliary display image update for a specific turn
    Called by Laura's agent during tool use loop
    """
    turn = request.get("turn")
    image = request.get("image")

    if not turn or not image:
        return {"status": "error", "message": "Missing turn or image"}

    # Send to auxiliary display via WebSocket
    if "auxiliary_display" in connections:
        ws = connections["auxiliary_display"]
        message = {
            "type": "display_image",
            "image": image
        }
        await ws.send_text(json.dumps(message))
        logger.info(f"🖥️ Sent auxiliary display image for turn {turn}: {image}")
        return {
            "status": "ok",
            "turn": turn,
            "image": image,
            "message": f"Auxiliary display will show '{image}' for turn {turn}"
        }
    else:
        logger.warning(f"⚠️ Auxiliary display not connected (turn {turn})")
        return {
            "status": "warning",
            "message": "Auxiliary display not connected",
            "turn": turn
        }


async def execute_tool(tool: Dict[str, Any]):
    """
    Execute a tool command from Laura
    Routes to Claude's device for star manipulation
    """
    action = tool.get("action")
    parameters = tool.get("parameters", {})

    # All actions are Claude star manipulations
    if "claude" in connections:
        message = {
            "type": "star_manipulation",
            "action": action,
            "parameters": parameters
        }
        await connections["claude"].send_text(json.dumps(message))
        logger.info(f"⭐ Sent {action} to Claude's star")
    else:
        logger.warning("⚠️ Claude not connected for tool execution")


def set_scene_callback(callback):
    """
    Register callback for playback_complete events
    Called by turn_manager to receive notifications
    """
    global active_scene_callback
    active_scene_callback = callback


if __name__ == "__main__":
    import os
    port = int(os.getenv("HUB_PORT", 8766))  # Default 8766 for gameboy demo (matches port forwarding)

    print(f"🎮 Starting Gameboy Hub on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
