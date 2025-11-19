# communication/mcp_session_manager.py
"""
Manages the connection, session initialization, and message sending
to the main MCP server.
"""

import json
import traceback
from datetime import datetime
from config.client_config import client_settings, get_active_tts_provider

class MCPSessionManager:
    def __init__(self):
        self.session_id: str | None = None
        self.mcp_session = None

    def set_session(self, session):
        self.mcp_session = session

    def clear_session(self):
        self.session_id = None
        self.mcp_session = None

    async def initialize_session(self, device_id: str) -> bool:
        """Initializes the client session with the MCP server."""
        if not self.mcp_session:
            print("[ERROR] MCP session object not available for registration.")
            return False
        
        try:
            print("[INFO] Performing MCP handshake with server...")
            await self.mcp_session.initialize()
            
            registration_payload = {
                "device_id": device_id,
                "capabilities": {
                    "input": ["text", "audio", "image"],
                    "output": ["text", "audio"],
                    "tts_mode": client_settings.get("tts_mode", "api"),
                    "api_tts_provider": get_active_tts_provider(),
                    "supports_caching": True
                }
            }
            
            response_obj = await self.mcp_session.call_tool("register_device", arguments=registration_payload)
            response_data = json.loads(response_obj.content[0].text)

            if isinstance(response_data, dict) and response_data.get("session_id"):
                self.session_id = response_data["session_id"]
                print(f"[INFO] Device registration successful. Session ID: {self.session_id}")
                return True
            else:
                print(f"[ERROR] Device registration failed. Response: {response_data}")
                return False
        except Exception as e:
            print(f"[ERROR] Error during session initialization: {e}")
            traceback.print_exc()
            return False

    async def send_to_server(self, transcript: str, wake_frame: str = None) -> dict | None:
        """Sends a transcript (and optional image) to the server."""
        if not self.session_id or not self.mcp_session:
            print("[ERROR] Session not initialized. Cannot send message.")
            return {"text": "Error: Client session not ready.", "mood": "error"}

        word_count = len(transcript.strip().split()) if transcript else 0
        if word_count <= 2:
            print(f"[INFO] Rejecting short transcript ({word_count} words): '{transcript}'")
            return None

        try:
            if wake_frame:
                messages_content = [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": wake_frame}},
                    {"type": "text", "text": transcript}
                ]
                tool_call_args = {
                    "session_id": self.session_id, "input_type": "multimodal",
                    "payload": {"messages": [{"role": "user", "content": messages_content}]},
                    "output_mode": ["text", "audio"], "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                print(f"[WAKE] Sending multimodal message with camera frame ({len(wake_frame)} chars) and transcript: '{transcript}'")
                print(f"[WAKE DEBUG] Payload structure: input_type={tool_call_args['input_type']}, messages[0].content length={len(messages_content)}")
            else:
                tool_call_args = {
                    "session_id": self.session_id, "input_type": "text",
                    "payload": {"text": transcript}, "output_mode": ["text", "audio"],
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }

            response_payload = await self.mcp_session.call_tool("run_LAURA", arguments=tool_call_args)

            if hasattr(response_payload, 'content') and response_payload.content:
                return json.loads(response_payload.content[0].text)
            elif isinstance(response_payload, dict):
                return response_payload
            else:
                return {"text": "Sorry, I received an unexpected response format.", "mood": "confused"}

        except Exception as e:
            print(f"[ERROR] Failed to call server: {e}")
            self.clear_session()
            return {"text": "Sorry, a communication problem occurred.", "mood": "error"}
