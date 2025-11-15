#!/usr/bin/env python3

# Suppress Jack audio server errors (must be before any imports that use audio)
import os
import sys
os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
os.environ['JACK_NO_START_SERVER'] = '1'

# Redirect stderr to suppress Jack library warnings
import contextlib
_stderr_backup = sys.stderr
_null = open(os.devnull, 'w')

def suppress_jack_errors():
    """Context manager to suppress Jack audio errors during PyAudio init"""
    sys.stderr = _null

def restore_stderr():
    """Restore stderr after suppression"""
    sys.stderr = _stderr_backup

# Suppress before any audio imports
suppress_jack_errors()

import asyncio
import json
import traceback
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style, init
from aiohttp import web
import aiohttp
import requests
from mutagen.mp3 import MP3

# Add snowboy to path for wake word detection
sys.path.insert(0, str(Path(__file__).parent / "snowboy"))

# MCP Imports
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# Local Component Imports
from system.audio_manager import AudioManager
from communication.client_tts_handler import TTSHandler
from speech_capture.vosk_websocket_adapter import VoskTranscriber
from display.display_manager import DisplayManager

# New Modular Components
from speech_capture.speech_processor import SpeechProcessor
from system.conversation_manager import ConversationManager
from system.input_manager import InputManager
from system.notification_manager import NotificationManager
from system.system_command_manager import SystemCommandManager
from system.audio_coordinator import AudioCoordinator

# Reachy Control Components
from daemon_client import DaemonClient
from moves import MovementManager
from camera_worker import CameraWorker
from head_wobbler import HeadWobbler
from yolo_head_tracker import HeadTracker
from daemon_media_wrapper import DaemonMediaWrapper
import mood_extractor

# Configuration and Utilities
from config.client_config import (
    SERVER_URL, DEVICE_ID, VOSK_MODEL_PATH, AUDIO_SAMPLE_RATE,
    client_settings, save_client_settings, get_active_tts_provider, set_active_tts_provider
)
from speech_capture.vosk_readiness_checker import vosk_readiness, ensure_vosk_ready

# Initialize colorama
init()

# Restore stderr after imports (Jack errors during PyAudio init are now suppressed)
restore_stderr()

# Sound assets base path
SOUND_BASE_PATH = Path("/home/user/more transfer/assets/sounds")


def get_random_audio(category: str, subtype: str = None):
    """Get random audio file for given category"""
    import random
    try:
        base_sound_dir = SOUND_BASE_PATH / "laura"
        
        if category == "wake" and subtype in ["Laura.pmdl", "Wake_up_Laura.pmdl", "GD_Laura.pmdl"]:
            context_map = {
                "Laura.pmdl": "standard",
                "Wake_up_Laura.pmdl": "sleepy", 
                "GD_Laura.pmdl": "frustrated"
            }
            folder = context_map.get(subtype, "standard")
            audio_path = Path(f"{base_sound_dir}/wake_sentences/{folder}")
        else:
            audio_path = Path(f"{base_sound_dir}/{category}_sentences")
            if subtype and (Path(f"{audio_path}/{subtype}")).exists():
                audio_path = Path(f"{audio_path}/{subtype}")
        
        audio_files = []
        if audio_path.exists():
            audio_files = list(audio_path.glob('*.mp3')) + list(audio_path.glob('*.wav'))
        
        if audio_files:
            return str(random.choice(audio_files))
        return None
    except Exception as e:
        print(f"Error in get_random_audio: {str(e)}")
        return None


class PiMCPClient:
    """
    Modular MCP Client for Pi 500 with clean separation of concerns.
    
    This orchestrator coordinates between all the specialized managers
    while maintaining a minimal footprint for the main client logic.
    """
    
    def __init__(self, server_url: str, device_id: str):
        self.server_url = server_url
        self.device_id = device_id
        self.session_id: str | None = None
        self.mcp_session: ClientSession | None = None
        
        # Initialize core components
        self.audio_manager = AudioManager(sample_rate=AUDIO_SAMPLE_RATE)
        self.tts_handler = TTSHandler()
        self.state_tracker = DisplayManager()  # Visual display with pygame

        # Initialize Reachy control components
        print("[INFO] Connecting to Reachy daemon...")
        self.daemon_client = DaemonClient("http://localhost:8100")
        self.media_manager = DaemonMediaWrapper("http://localhost:8100")
        self.head_tracker = HeadTracker(device="cpu")  # Use "cuda" if Pi has GPU

        # Initialize camera worker (30Hz face tracking)
        self.camera_worker = CameraWorker(
            media_manager=self.media_manager,
            head_tracker=self.head_tracker,
            daemon_client=self.daemon_client,
            debug_window=False  # Disabled - no debug window
        )

        # Initialize movement manager (100Hz control loop)
        self.movement_manager = MovementManager(
            current_robot=self.daemon_client,
            camera_worker=self.camera_worker
        )

        # Link movement_manager back to camera_worker for breathing synchronization
        self.camera_worker.movement_manager = self.movement_manager

        # Initialize head wobbler (speech sway)
        self.head_wobbler = HeadWobbler(
            set_speech_offsets=self.movement_manager.set_speech_offsets
        )

        # Register state callback for face tracking control
        self.state_tracker.register_callback(self._on_state_change)

        # Mood tracking for TTS coordination
        self._pending_mood = None
        self._mood_loop_running = False  # Prevent multiple mood loops from spawning
        self._mood_process = None  # Track active mood subprocess for cleanup
        self._mood_reject_count = 0  # Kill process if spam exceeds limit

        self.transcriber = VoskTranscriber(sample_rate=AUDIO_SAMPLE_RATE)
        
        # Initialize specialized managers
        self.input_manager = InputManager(self.audio_manager)
        self.audio_coordinator = AudioCoordinator(self.audio_manager)
        self.speech_processor = SpeechProcessor(
            self.audio_manager, 
            self.transcriber, 
            None  # Will be set after keyboard initialization
        )
        self.conversation_manager = ConversationManager(
            self.speech_processor,
            self.audio_coordinator,
            self.tts_handler,
            client_settings
        )
        self.notification_manager = NotificationManager(
            self.audio_coordinator,
            self.tts_handler
        )
        self.system_command_manager = SystemCommandManager(
            client_settings,
            save_client_settings,
            get_active_tts_provider,
            set_active_tts_provider
        )
        
        # Provide notification manager reference for testing commands
        self.system_command_manager._notification_manager = self.notification_manager
        
        # Initialize keyboard
        self.input_manager.initialize_keyboard()
        self.speech_processor.keyboard_device = self.input_manager.keyboard_device

    def _on_state_change(self, new_state, old_state, mood, text):
        """Callback for state changes - controls face tracking and mood animations.

        Face tracking is ONLY active during idle/breathing state.
        Disabled during listening, thinking, and speaking.
        Mood animations are triggered when entering speaking state.
        """
        # Enable face tracking only during idle state
        if new_state == "idle":
            self.camera_worker.set_head_tracking_enabled(True)
            print("[INFO] Face tracking ENABLED (idle/breathing)")
        else:
            # Disable during all other states (listening, thinking, speaking, sleep, etc.)
            self.camera_worker.set_head_tracking_enabled(False)
            print(f"[INFO] Face tracking DISABLED ({new_state})")

        # Trigger mood animations when entering speaking state
        if new_state == "speaking" and self._pending_mood:
            mood_to_play = self._pending_mood
            print(f"[INFO] Triggering mood animation: {mood_to_play}")

            # Run mood animation in background thread so it doesn't block
            import threading
            mood_thread = threading.Thread(
                target=mood_extractor.play_mood_loop_debug,
                args=(mood_to_play,),
                daemon=True
            )
            mood_thread.start()

            # Clear pending mood after triggering
            self._pending_mood = None

    def start_reachy_threads(self):
        """Start Reachy control threads (MovementManager, CameraWorker)."""
        print("[INFO] Starting Reachy control threads...")
        self.camera_worker.start()
        self.movement_manager.start()
        print("[INFO] Reachy control threads started")

    def stop_reachy_threads(self):
        """Stop Reachy control threads."""
        print("[INFO] Stopping Reachy control threads...")
        self.movement_manager.stop()
        self.camera_worker.stop()
        print("[INFO] Reachy control threads stopped")

    async def initialize_session(self):
        """Initialize the client session with the MCP server"""
        try:
            if not self.mcp_session:
                print("[ERROR] MCP session object not available for registration.")
                return False
                
            print("[INFO] Performing MCP handshake with server...")
            await self.mcp_session.initialize()
            print("[INFO] MCP handshake completed successfully.")
            
            await asyncio.sleep(2.0)  # Give server time to be ready

            registration_payload = {
                "device_id": self.device_id,
                "capabilities": {
                    "input": ["text", "audio"],
                    "output": ["text", "audio"],
                    "tts_mode": client_settings.get("tts_mode", "api"),
                    "api_tts_provider": get_active_tts_provider(),
                    "supports_caching": True
                }
            }
            
            print(f"[INFO] Calling 'register_device' tool with payload: {registration_payload}")
            response_obj = await self.mcp_session.call_tool("register_device", arguments=registration_payload)

            if hasattr(response_obj, 'content') and response_obj.content:
                text_content = response_obj.content[0].text
                response_data = json.loads(text_content)
            else:
                response_data = response_obj

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

    async def send_to_server(self, transcript: str) -> dict | None:
        """Send text to MCP server, forward response to Reachy for mood movements + TTS"""
        if not self.session_id or not self.mcp_session:
            print("[ERROR] Session not initialized. Cannot send message.")
            return {"text": "Error: Client session not ready.", "mood": "error"}

        # Filter out short transcripts (2 words or less)
        word_count = len(transcript.strip().split())
        if word_count <= 2:
            print(f"[INFO] Rejecting short transcript ({word_count} word{'s' if word_count != 1 else ''}): '{transcript}'")
            return None

        try:
            tool_call_args = {
                "session_id": self.session_id,
                "input_type": "text",
                "payload": {"text": transcript},
                "output_mode": ["text", "audio"],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            print(f"[INFO] Calling 'run_LAURA' tool...")
            response_payload = await self.mcp_session.call_tool("run_LAURA", arguments=tool_call_args)

            # Parse response properly
            parsed_response = None
            if hasattr(response_payload, 'content') and response_payload.content:
                json_str = response_payload.content[0].text
                parsed_response = json.loads(json_str)
            elif isinstance(response_payload, dict):
                parsed_response = response_payload
            else:
                print(f"[ERROR] Unexpected response format: {type(response_payload)}")
                return {"text": "Sorry, I received an unexpected response format.", "mood": "confused"}

            if isinstance(parsed_response, dict) and "text" in parsed_response:
                # Extract mood and trigger local Reachy movements
                response_text = parsed_response["text"]
                mood = mood_extractor.extract_mood_marker(response_text)

                if mood:
                    print(f"[INFO] Mood detected: {mood}")
                    # Mood movements will play during TTS (handled by mood_extractor in speaking state)
                    # Store mood for later use in speaking state
                    self._pending_mood = mood
                else:
                    print("[INFO] No mood marker detected in response")
                    self._pending_mood = None

                return parsed_response
            else:
                print(f"[ERROR] Invalid response: {parsed_response}")
                return {"text": "Sorry, I received an unexpected response.", "mood": "confused"}

        except (ConnectionError, ConnectionRefusedError, OSError) as e:
            print(f"[ERROR] Connection lost during server call: {e}")
            # Clear session to trigger reconnection
            self.mcp_session = None
            self.session_id = None
            return {"text": "Connection lost. Reconnecting...", "mood": "error"}
        except Exception as e:
            print(f"[ERROR] Failed to call server: {e}")
            traceback.print_exc()
            return {"text": "Sorry, a communication problem occurred.", "mood": "error"}

    async def run_main_loop(self):
        """Main interaction loop with natural conversation flow"""
        print("[INFO] Main interaction loop started.")

        while True:
            try:
                current_state = self.state_tracker.get_state()

                # Handle Claude Code plugin mood state
                if current_state == 'cc_plugin_mood':
                    # Check if pending_mood exists - if None, already processed
                    if not hasattr(self.state_tracker, 'pending_mood') or self.state_tracker.pending_mood is None:
                        self.state_tracker.update_state("idle")
                        continue

                    # Check if a mood loop is already running - prevent spawn loop crash
                    if self._mood_loop_running:
                        self._mood_reject_count += 1
                        if self._mood_reject_count >= 5:
                            print("[MOOD STATE] Spam limit exceeded (5 rejections), killing process")
                            import os
                            os._exit(1)
                        self.state_tracker.update_state("idle")
                        continue

                    mood = getattr(self.state_tracker, 'pending_mood', 'thoughtful')

                    # CRITICAL: Clear pending_mood immediately to prevent re-processing
                    self.state_tracker.pending_mood = None

                    print(f"[MOOD STATE] Starting mood in background: {mood}")

                    # Reset reject counter on successful start
                    self._mood_reject_count = 0

                    # Set flag BEFORE spawning to prevent race condition
                    self._mood_loop_running = True

                    # Run mood_extractor.py as background subprocess
                    import subprocess
                    import sys
                    import threading

                    def run_mood_subprocess():
                        """Run mood subprocess and clear flag when done"""
                        process = None
                        try:
                            # Pause breathing to prevent race condition
                            print("[MOOD STATE] Pausing breathing for mood coordination")
                            self.movement_manager.pause_breathing()

                            # Prepare input for mood_extractor (it reads from stdin)
                            mood_input = f"<!-- MOOD: {mood} -->\nTriggered via plugin"

                            # Run mood_extractor.py in background (non-blocking)
                            mood_script = "/home/user/reachy/pi_reachy_deployment/mood_extractor.py"
                            process = subprocess.Popen(
                                [sys.executable, mood_script],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=None,  # Allow debug output to print directly to console
                                text=True
                            )

                            # Store process reference for cleanup
                            self._mood_process = process

                            # Send input and wait for completion
                            stdout, _ = process.communicate(input=mood_input, timeout=65)

                            if process.returncode == 0:
                                print(f"[MOOD STATE] Mood loop completed successfully")
                            else:
                                print(f"[MOOD STATE] Mood loop exited with code {process.returncode}")

                        except subprocess.TimeoutExpired:
                            print(f"[MOOD STATE] Mood loop timed out (safety timeout)")
                            if process:
                                process.kill()
                                process.wait()  # Clean up zombie
                        except Exception as e:
                            print(f"[MOOD STATE] Mood execution error: {e}")
                        finally:
                            # Clear process reference
                            self._mood_process = None

                            # Resume breathing after mood completes
                            print("[MOOD STATE] Resuming breathing")
                            self.movement_manager.resume_breathing()

                            # Always clear the flag when done
                            self._mood_loop_running = False
                            print(f"[MOOD STATE] Mood loop finished, flag cleared")

                    # Start mood thread
                    mood_thread = threading.Thread(target=run_mood_subprocess, daemon=True)
                    mood_thread.start()

                    # Return to idle state immediately (mood runs in background)
                    self.state_tracker.update_state("idle")
                    self.input_manager.update_last_interaction()
                    print("[MOOD STATE] Returned to idle state (mood running in background)")
                    continue

                # Only check for wake events during sleep/idle/code
                if current_state in ['sleep', 'idle', 'code']:
                    # Check for wake events
                    wake_event_source = await self.input_manager.check_for_wake_events()
                    
                    # If no wake event, continue monitoring
                    if not wake_event_source:
                        await asyncio.sleep(0.05)  # Optimized for faster wake word detection
                        continue
                    
                    # CRITICAL: Release microphone from wake word detection immediately
                    if 'wakeword' in wake_event_source:
                        print("[DEBUG] Releasing microphone from wake word detection for voice input")
                        self.input_manager.stop_wake_word_detection()
                    
                    # Handle wake event - play contextual audio efficiently
                    if 'wakeword' in wake_event_source:
                        model_name = wake_event_source.split('(')[1].rstrip(')')
                        print(f"[DEBUG] Extracted model_name: '{model_name}'")
                        
                        # Check if this is a medicine acknowledgment wake word
                        if model_name == "tookmycrazypills.pmdl":
                            print("[INFO] Medicine taken acknowledgment via wake word")
                            # Clear the medicine reminder
                            await self.system_command_manager.system_manager.clear_reminder("medicine", self.mcp_session, self.session_id)
                            # Play success sound (same as startup)
                            success_sound = str(SOUND_BASE_PATH / "sound_effects" / "successfulloadup.mp3")
                            if os.path.exists(success_sound):
                                await self.audio_coordinator.play_audio_file(success_sound)
                            self.state_tracker.update_state("idle")
                            continue  # Skip everything else
                        
                    # Switch persona (display + voice) FIRST based on wake source
                    if wake_event_source == "keyboard_laura" or (wake_event_source.startswith("wakeword") and not self._should_route_to_claude_code(wake_event_source)):
                        # Switch to LAURA persona
                        self._switch_to_persona('laura')
                    elif wake_event_source == "keyboard_code" or self._should_route_to_claude_code(wake_event_source):
                        # Switch to Claude Code persona
                        self._switch_to_persona('claude_code')
                    
                    # Initialize audio before updating display
                    await self.audio_manager.initialize_input()
                    
                    # Now update display - mic is ready and persona is correct
                    self.state_tracker.update_state('listening')
                    
                    # Play appropriate wake audio AFTER display update
                    if wake_event_source.startswith("wakeword"):
                        model_name = wake_event_source.split()[1].strip('()')
                        
                        if model_name == "claudecode.pmdl":
                            # Play radar ping for Claude Code
                            radar_ping = str(SOUND_BASE_PATH / "sound_effects" / "radarping.mp3")
                            if os.path.exists(radar_ping):
                                print("[DEBUG] Playing Claude Code radar ping...")
                                await self.audio_coordinator.play_audio_file(radar_ping)
                        else:
                            # Play normal wake audio for LAURA
                            wake_audio = get_random_audio('wake', model_name)
                            print(f"[DEBUG] get_random_audio returned: {wake_audio}")
                            if wake_audio:
                                print(f"[DEBUG] About to play wake audio: {wake_audio}")
                                await self.audio_coordinator.play_audio_file(wake_audio)
                            else:
                                print(f"[DEBUG] No wake audio returned for model: {model_name}")
                    elif wake_event_source == "keyboard_code":
                        # Play radar ping for keyboard-triggered Claude Code
                        radar_ping = str(SOUND_BASE_PATH / "sound_effects" / "radarping.mp3")
                        if os.path.exists(radar_ping):
                            print("[DEBUG] Playing Claude Code radar ping (keyboard triggered)...")
                            await self.audio_coordinator.play_audio_file(radar_ping)
                    
                    # Capture speech - use push-to-talk mode for keyboard, VAD for wakeword
                    if wake_event_source in ["keyboard_laura", "keyboard_code"]:
                        print("[INFO] Using push-to-talk mode (no VAD timeouts)")
                        # Use push-to-talk without callback - we'll play teletype after capture
                        transcript = await self.speech_processor.capture_speech_push_to_talk(self.state_tracker)
                        
                        # Set cooldown to prevent immediate re-trigger from held keys
                        self.input_manager.set_keyboard_cooldown(0.2)
                        
                        # Teletype sound will play when typing actually begins (moved to injection function)
                    else:
                        # Use faster 2-second silence for Claude Code, normal 3.5s for LAURA
                        if self._should_route_to_claude_code(wake_event_source):
                            print("[INFO] Using VAD mode with 2s Claude Code timeout")
                            # For Claude Code, play confirmation sound immediately on silence detection
                            transcript = await self.speech_processor.capture_speech_with_unified_vad(
                                self.state_tracker, is_follow_up=False, claude_code_mode=True,
                                immediate_feedback_callback=self._play_claude_code_confirmation
                            )
                        else:
                            print("[INFO] Using VAD mode with standard timeouts")
                            transcript = await self.speech_processor.capture_speech_with_unified_vad(self.state_tracker, is_follow_up=False)
                    
                    if not transcript:
                        print("[INFO] No speech detected, returning to previous state")
                        # Restart wake word detection
                        self.input_manager.restart_wake_word_detection()
                        # Determine the appropriate return state based on context
                        if current_state == "code":
                            return_state = "code"
                        elif self.state_tracker.display_profile == 'claude_code':
                            # If we're in Claude Code profile but not in code state, go to idle
                            return_state = "idle"
                        else:
                            return_state = "idle"
                        self.state_tracker.update_state(return_state)
                        continue
                    
                    # Update interaction time on successful speech capture
                    self.input_manager.update_last_interaction()
                    
                    # In code mode, inject everything to Claude Code (except system commands)
                    if current_state == "code":
                        print(f"[INFO] Code mode active - injecting to Claude Code: '{transcript}'")
                        self.state_tracker.update_state('thinking')
                        await self.inject_to_claude_code(transcript)
                        self.state_tracker.update_state('code')
                        continue
                    
                    # Direct injection to Claude Code if SHIFT+Left Meta was used OR specific wake word
                    if wake_event_source == "keyboard_code" or self._should_route_to_claude_code(wake_event_source):
                        print(f"[INFO] Injecting directly to Claude Code: '{transcript}'")
                        
                        self.state_tracker.update_state('thinking')
                        
                        # Data processing sound already started by immediate feedback
                        # Just inject with phase transitions
                        await self.inject_to_claude_code_with_sounds(transcript, None)
                        
                        # Reset any lingering conversation state after Claude Code
                        self.conversation_manager.reset_conversation_state()
                        
                        self.state_tracker.update_state('idle')
                        continue
                    
                    # Check for note transfer wake word
                    if self._should_send_note_to_mac(wake_event_source):
                        print(f"[INFO] Note transfer wake word detected - sending pi500_note.txt to Mac")
                        self.state_tracker.update_state('thinking')
                        await self.send_note_to_mac()
                        self.state_tracker.update_state('idle')
                        continue
                    
                    # Check for send Enter key wake word
                    if self._should_send_enter_key(wake_event_source):
                        print(f"[INFO] Send Enter wake word detected - sending Enter key")
                        await self.send_enter_key()
                        continue
                    
                    # Check for system commands
                    is_cmd, cmd_type, cmd_arg = self.system_command_manager.detect_system_command(transcript)
                    if is_cmd:
                        await self.system_command_manager.handle_system_command(
                            cmd_type, cmd_arg, self.mcp_session,
                            self.tts_handler, self.audio_coordinator, self.state_tracker
                        )
                        self.state_tracker.update_state('idle')
                        continue
                    
                    # Check for document uploads
                    await self.system_command_manager.check_and_upload_documents(self.mcp_session, self.session_id)
                    
                    # Process normal conversation
                    self.state_tracker.update_state('thinking')
                    response = await self.send_to_server(transcript)
                    
                    # Only process response if not filtered out
                    if response is not None:
                        # Handle response through conversation manager
                        await self.conversation_manager.process_initial_response(
                            response, self.state_tracker, self, self.system_command_manager
                        )
                    else:
                        # Transcript was too short, return to idle
                        self.state_tracker.update_state("idle")
                        
                else:
                    # In other states, brief sleep
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"[ERROR] Error in main loop: {e}")
                traceback.print_exc()
                self.state_tracker.update_state("error", text="System Error")
                await asyncio.sleep(2)
                # Return to appropriate state based on current mode
                return_state = "code" if self.state_tracker.get_state() == "code" else "idle"
                self.state_tracker.update_state(return_state)

    async def handle_tts_conversation(self, request):
        """TTS endpoint that returns to idle state - for questions/confirmations"""
        try:
            data = await request.json()
            text = data.get('text', '')
            mood = data.get('mood', 'explaining')
            
            if text:
                # Update display to speaking with appropriate mood
                self.state_tracker.update_state("speaking", mood=mood)
                
                # Send to TTS server
                try:
                    response = requests.post(
                        "http://localhost:5001/tts",
                        headers={"Content-Type": "application/json"},
                        json={"text": text, "voice": "claude"},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        # Get exact audio duration using mutagen
                        response_data = response.json()
                        audio_file_path = response_data.get('audio_file')
                        
                        if audio_file_path and os.path.exists(audio_file_path):
                            try:
                                audio_info = MP3(audio_file_path)
                                exact_duration = audio_info.info.length
                                print(f"[TTS Conversation] Exact audio duration: {exact_duration:.2f}s")
                                await asyncio.sleep(exact_duration)
                            except Exception as mutagen_error:
                                print(f"[TTS Conversation] Mutagen error: {mutagen_error}, falling back to estimation")
                                # Fallback to estimation
                                word_count = len(text.split())
                                estimated_duration = max(2, (word_count / 150) * 60)
                                await asyncio.sleep(estimated_duration)
                        else:
                            print(f"[TTS Conversation] No audio file path returned, using estimation")
                            # Fallback to estimation  
                            word_count = len(text.split())
                            estimated_duration = max(2, (word_count / 150) * 60)
                            await asyncio.sleep(estimated_duration)
                    
                except Exception as e:
                    print(f"[TTS Error] {e}")
                
                # Return to idle state - waiting for user input
                self.input_manager.restart_wake_word_detection()
                self.state_tracker.update_state("idle")
                
            return web.json_response({"status": "success", "state": "idle"})
        except Exception as e:
            print(f"[TTS Conversation Error] {e}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)
    
    async def handle_tts_working(self, request):
        """TTS endpoint that goes to execution state - for status updates while working"""
        try:
            data = await request.json()
            text = data.get('text', '')
            mood = data.get('mood', 'solution')

            if text:
                # Update display to speaking with appropriate mood
                self.state_tracker.update_state("speaking", mood=mood)

                # Send to TTS server
                try:
                    response = requests.post(
                        "http://localhost:5001/tts",
                        headers={"Content-Type": "application/json"},
                        json={"text": text, "voice": "claude"},
                        timeout=30
                    )

                    if response.status_code == 200:
                        # Get exact audio duration using mutagen
                        response_data = response.json()
                        audio_file_path = response_data.get('audio_file')

                        if audio_file_path and os.path.exists(audio_file_path):
                            try:
                                audio_info = MP3(audio_file_path)
                                exact_duration = audio_info.info.length
                                print(f"[TTS Working] Exact audio duration: {exact_duration:.2f}s")
                                await asyncio.sleep(exact_duration)
                            except Exception as mutagen_error:
                                print(f"[TTS Working] Mutagen error: {mutagen_error}, falling back to estimation")
                                # Fallback to estimation
                                word_count = len(text.split())
                                estimated_duration = max(2, (word_count / 150) * 60)
                                await asyncio.sleep(estimated_duration)
                        else:
                            print(f"[TTS Working] No audio file path returned, using estimation")
                            # Fallback to estimation
                            word_count = len(text.split())
                            estimated_duration = max(2, (word_count / 150) * 60)
                            await asyncio.sleep(estimated_duration)

                except Exception as e:
                    print(f"[TTS Error] {e}")

                # Return to execution state - still working
                self.state_tracker.update_state("execution")

            return web.json_response({"status": "success", "state": "execution"})
        except Exception as e:
            print(f"[TTS Working Error] {e}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def handle_display_update(self, request):
        """API endpoint to update display from external services (gameboy_hub)"""
        try:
            data = await request.json()
            state = data.get('state', 'idle')
            mood = data.get('mood')
            text = data.get('text')

            print(f"[API] Display update request: state={state}, mood={mood}")

            # Update display
            self.state_tracker.update_state(state, mood=mood, text=text)

            return web.json_response({
                "status": "success",
                "state": state,
                "mood": mood
            })
        except Exception as e:
            print(f"[API Error] Display update failed: {e}")
            return web.json_response({
                "status": "error",
                "error": str(e)
            }, status=500)

    async def handle_mood_trigger(self, request):
        """Trigger mood-based movements via Claude Code plugin"""
        try:
            data = await request.json()
            mood = data.get('mood', '')

            if not mood:
                return web.json_response({
                    "status": "error",
                    "error": "No mood specified"
                }, status=400)

            # Store mood in state tracker for main loop to process
            self.state_tracker.pending_mood = mood
            self.state_tracker.update_state("cc_plugin_mood")

            print(f"[MOOD API] Triggered mood: {mood}")

            return web.json_response({
                "status": "success",
                "mood": mood
            })
        except Exception as e:
            print(f"[API Error] Mood trigger failed: {e}")
            return web.json_response({
                "status": "error",
                "error": str(e)
            }, status=500)

    async def start_tts_server(self):
        """Start the HTTP server for TTS endpoints"""
        app = web.Application()
        app.router.add_post('/tts/conversation', self.handle_tts_conversation)
        app.router.add_post('/tts/working', self.handle_tts_working)
        app.router.add_post('/display/update', self.handle_display_update)
        app.router.add_post('/mood/trigger', self.handle_mood_trigger)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8888)
        await site.start()
        print("[INFO] TTS HTTP endpoints started on localhost:8888")
        print("  - /tts/conversation: Returns to idle state")  
        print("  - /tts/working: Returns to execution state")
        return runner
    
    async def run(self):
        """Main client run loop with multi-task architecture"""
        print(f"{Fore.CYAN}PiMCPClient v2 run loop started.{Fore.WHITE}")
        self.state_tracker.update_state("boot")
        
        # TTS HTTP server enabled for display API control
        tts_server = await self.start_tts_server()

        # Start Reachy control threads (background)
        self.start_reachy_threads()

        # Start background tasks
        background_tasks = [
            asyncio.create_task(self.notification_manager.check_for_notifications_loop(
                self.mcp_session, self.session_id, self.state_tracker
            )),
            asyncio.create_task(self.sleep_timeout_monitor()),
        ]

        connection_attempts = 0
        handshake_failures = 0
        connection_failures = 0  # Track connection failures separately
        code_mode_active = False
        
        while True:
            try:
                # If in code mode, handle it differently
                if code_mode_active:
                    # Run main loop without server connection
                    print("[INFO] Running in code mode - server connection bypassed")
                    self.state_tracker.update_state("code")
                    
                    # Add main loop task and run it
                    main_loop_task = asyncio.create_task(self.run_main_loop())
                    all_tasks = background_tasks + [main_loop_task]
                    
                    try:
                        await asyncio.gather(*all_tasks, return_exceptions=True)
                    except Exception as e:
                        print(f"[ERROR] Task execution error in code mode: {e}")
                    finally:
                        # Cancel main loop task
                        main_loop_task.cancel()
                        try:
                            await main_loop_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Check if user wants to try reconnecting (wait a bit)
                    await asyncio.sleep(10)
                    continue
                
                connection_attempts += 1
                if connection_attempts > 1:
                    print(f"[INFO] Reconnection attempt #{connection_attempts - 1}")
                print(f"[INFO] Attempting to connect to MCP server at {self.server_url}...")
                
                async with sse_client(f"{self.server_url}/events/sse", headers={}) as (read, write):
                    print("[INFO] SSE client connected. Creating ClientSession...")
                    connection_attempts = 0  # Reset counter on successful connection
                    
                    async with ClientSession(read, write) as session:
                        self.mcp_session = session
                        print("[INFO] ClientSession active.")

                        if not await self.initialize_session():
                            print("[ERROR] Failed to initialize session. Reconnecting...")
                            handshake_failures += 1
                            raise Exception("Session initialization failed")

                        # Reset failures on successful connection
                        handshake_failures = 0
                        connection_failures = 0
                        code_mode_active = False
                        
                        self.state_tracker.update_state("idle")
                        print(f"{Fore.CYAN}✓ Session initialized successfully{Fore.WHITE}")
                        
                        # Play startup sound and transition to sleep (only on first connection)
                        if connection_attempts == 0:
                            print(f"\n{Fore.CYAN}=== Startup Sequence ==={Fore.WHITE}")
                            startup_sound = str(SOUND_BASE_PATH / "sound_effects" / "successfulloadup.mp3")
                            if os.path.exists(startup_sound):
                                try:
                                    print(f"{Fore.CYAN}Playing startup audio...{Fore.WHITE}")
                                    # Use idle for Claude Code profile, sleep for LAURA
                                    final_state = 'idle' if self.state_tracker.display_profile == 'claude_code' else 'sleep'
                                    self.state_tracker.update_state(final_state)
                                    await self.audio_coordinator.play_audio_file(startup_sound)
                                    print(f"{Fore.GREEN}✓ Startup audio complete{Fore.WHITE}")
                                except Exception as e:
                                    print(f"{Fore.YELLOW}Warning: Could not play startup sound: {e}{Fore.WHITE}")
                        else:
                            # Reconnection success - brief audio notification
                            print(f"{Fore.GREEN}✓ Reconnected to MCP server{Fore.WHITE}")
                            # Use idle for Claude Code profile, sleep for LAURA
                            final_state = 'idle' if self.state_tracker.display_profile == 'claude_code' else 'sleep'
                            self.state_tracker.update_state(final_state)
                        
                        print(f"{Fore.MAGENTA}🎧 Listening for wake word or press Raspberry button to begin...{Fore.WHITE}")
                        
                        # Update notification manager with session info
                        background_tasks[0].cancel()
                        background_tasks[0] = asyncio.create_task(
                            self.notification_manager.check_for_notifications_loop(
                                self.mcp_session, self.session_id, self.state_tracker
                            )
                        )
                        
                        # Add main loop to tasks and run all concurrently
                        main_loop_task = asyncio.create_task(self.run_main_loop())
                        all_tasks = background_tasks + [main_loop_task]
                        
                        try:
                            await asyncio.gather(*all_tasks, return_exceptions=True)
                        except Exception as e:
                            print(f"[ERROR] Task execution error: {e}")
                        finally:
                            # Cancel main loop task when connection ends
                            main_loop_task.cancel()
                            try:
                                await main_loop_task
                            except asyncio.CancelledError:
                                pass
                                
            except asyncio.CancelledError:
                print("[INFO] Main loop cancelled.")
                break
            except (ConnectionRefusedError, ConnectionError, OSError) as e:
                print(f"[ERROR] Connection failed: {e}. Server may be down.")
                connection_failures += 1
                
                # Check if we should enter code mode after 2 connection failures
                if connection_failures >= 2 and not code_mode_active:
                    print(f"[INFO] {connection_failures} connection failures detected. Entering code mode...")
                    self.state_tracker.update_state("code")
                    print("[INFO] Code mode active - speech will be routed to Claude Code")
                    code_mode_active = True
                else:
                    self.state_tracker.update_state("error", text="Server Offline")
                
                if not code_mode_active:
                    print(f"[INFO] Retrying connection in 30 seconds...")
                    await asyncio.sleep(30)
                else:
                    print("[INFO] Code mode active - stopping connection attempts. Use voice commands or keyboard.")
                    await asyncio.sleep(5)  # Short sleep before trying again to see if user wants to exit code mode
            except Exception as e:
                print(f"[ERROR] Unhandled connection-level exception: {e}")
                traceback.print_exc()
                connection_failures += 1  # Also count general exceptions as connection failures
                
                # Check if we should enter code mode after 2 connection failures
                if connection_failures >= 2 and not code_mode_active:
                    print(f"[INFO] {connection_failures} connection failures detected. Entering code mode...")
                    self.state_tracker.update_state("code")
                    print("[INFO] Code mode active - speech will be routed to Claude Code")
                    code_mode_active = True
                else:
                    self.state_tracker.update_state("error", text="Connection Error")
                
                if not code_mode_active:
                    print(f"[INFO] Retrying connection in 30 seconds...")
                    await asyncio.sleep(30)
                else:
                    print("[INFO] Code mode active - stopping connection attempts. Use voice commands or keyboard.")
                    await asyncio.sleep(5)  # Short sleep before trying again to see if user wants to exit code mode
            finally:
                self.mcp_session = None
                if connection_attempts == 0:  # Only show disconnected state if we were previously connected
                    self.state_tracker.update_state("disconnected")
        
        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def sleep_timeout_monitor(self):
        """Background task to monitor for sleep timeout (5 minutes of inactivity)"""
        SLEEP_TIMEOUT = 300  # 5 minutes in seconds
        
        while True:
            try:
                current_state = self.state_tracker.get_state()
                
                # Only check for sleep timeout when in idle state
                if current_state == 'idle':
                    time_since_interaction = self.input_manager.get_time_since_last_interaction()
                    
                    if time_since_interaction >= SLEEP_TIMEOUT:
                        print(f"[INFO] Sleep timeout reached ({time_since_interaction:.1f}s since last interaction)")
                        self.state_tracker.update_state('sleep')
                    
                await asyncio.sleep(1)  # Check every second for precision
                
            except Exception as e:
                print(f"[ERROR] Sleep timeout monitor error: {e}")
                await asyncio.sleep(5)  # Wait longer on error
            
    async def cleanup(self):
        """Clean up resources"""
        print("[INFO] Starting client cleanup...")

        # Kill any running mood subprocess to prevent task destruction errors
        if self._mood_process is not None:
            print("[INFO] Killing running mood subprocess...")
            try:
                self._mood_process.kill()
                self._mood_process.wait(timeout=2)
                print("[INFO] Mood subprocess killed")
            except Exception as e:
                print(f"[WARNING] Error killing mood subprocess: {e}")

        # Stop Reachy control threads
        self.stop_reachy_threads()

        if self.audio_coordinator:
            await self.audio_coordinator.cleanup()
        if self.audio_manager:
            await self.audio_manager.cleanup()
        if self.state_tracker:
            self.state_tracker.cleanup()
        if self.input_manager:
            self.input_manager.cleanup()
        print("[INFO] Client cleanup finished.")
    
    async def route_to_claude_code(self, transcript: str):
        """Route speech transcript to Claude Code with health check and session management"""
        from claude.claude_code_healthcheck import execute_claude_code_with_health_check
        
        try:
            print(f"[INFO] Routing to Claude Code: '{transcript}'")
            
            # Update display to show processing
            self.state_tracker.update_state("thinking", mood="focused")
            
            # Process with Claude Code using health check and session management
            result = await execute_claude_code_with_health_check(transcript)
            
            if result["success"]:
                response = result.get("response", "")
                execution_time = result.get("execution_time", 0)
                session_info = result.get("session_info", "Unknown session")
                
                print(f"[INFO] Claude Code completed in {execution_time:.1f}s via {session_info}")
                print(f"[INFO] Response: {response[:100]}{'...' if len(response) > 100 else ''}")
                
                # Determine if we should speak the response
                should_speak = self._should_speak_claude_response(response, transcript)
                
                if should_speak and response:
                    # Speak the response via TTS
                    self.state_tracker.update_state("speaking", mood="helpful")
                    
                    # Use a shorter summary for very long responses
                    if len(response) > 300:
                        speech_text = "Task completed successfully. Check the output for details."
                    else:
                        speech_text = response
                        
                    try:
                        # Use the existing TTS system
                        audio_file = await self.tts_handler.synthesize_speech(
                            speech_text,
                            voice_id="default"
                        )
                        
                        if audio_file:
                            await self.audio_manager.play_audio(audio_file)
                    except Exception as tts_error:
                        print(f"[ERROR] TTS failed: {tts_error}")
                        # Fall back to success sound
                        await self.audio_manager.play_audio(
                            str(SOUND_BASE_PATH / "sound_effects" / "successfulloadup.mp3")
                        )
                else:
                    # Just confirm completion without speaking the full response
                    await self.audio_manager.play_audio(
                        str(SOUND_BASE_PATH / "sound_effects" / "successfulloadup.mp3")
                    )
                    
            else:
                # Handle error
                error = result.get("error", "Unknown error")
                print(f"[ERROR] Claude Code failed: {error}")
                
                # Speak error notification
                self.state_tracker.update_state("speaking", mood="confused")
                try:
                    error_text = f"Claude Code error: {error}"
                    audio_file = await self.tts_handler.synthesize_speech(
                        error_text[:200],  # Limit error message length
                        voice_id="default"
                    )
                    if audio_file:
                        await self.audio_manager.play_audio(audio_file)
                except:
                    # Fall back to error sound if TTS fails
                    pass
                    
        except Exception as e:
            print(f"[ERROR] Failed to route to Claude Code: {e}")
            # Handle error gracefully
            
        finally:
            # Update interaction time on successful completion
            self.input_manager.update_last_interaction()
            # Return to idle state
            self.state_tracker.update_state("idle", mood="casual")
    
    def _should_speak_claude_response(self, response: str, original_command: str) -> bool:
        """Determine if Claude Code response should be spoken"""
        if not response:
            return False
            
        # Don't speak very long responses
        if len(response) > 500:
            return False
            
        # Don't speak responses that look like code
        code_indicators = ['```', 'def ', 'class ', 'import ', 'function', '{', '}', 'const ', 'let ', 'var ']
        if any(indicator in response for indicator in code_indicators):
            return False
            
        # Don't speak file paths or technical output
        if response.startswith('/') or 'http://' in response or 'https://' in response:
            return False
            
        # Check for coding-related commands
        coding_keywords = ['create', 'write', 'implement', 'code', 'function', 'debug', 'fix', 'refactor']
        is_coding_command = any(keyword in original_command.lower() for keyword in coding_keywords)
        
        # For coding commands, be more conservative about speaking
        if is_coding_command and len(response) > 200:
            return False
            
        # Speak conversational responses
        return True
    
    async def inject_to_claude_code(self, transcript: str):
        """Inject transcript directly into Claude Code terminal using virtual keyboard"""
        await self.inject_to_claude_code_with_sounds(transcript, None)
    
    async def inject_to_claude_code_with_sounds(self, transcript: str, data_processing_task):
        """Inject transcript with phase-based sound transitions"""
        import subprocess
        import os
        
        try:
            print(f"[INFO] Injecting to Claude Code: '{transcript}'")
            
            # Path to our voice injector
            project_root = os.path.dirname(os.path.abspath(__file__))
            injector_script = os.path.join(project_root, "claude", "claude_voice_injector.py")
            venv_python = os.path.join(project_root, "venv", "bin", "python")
            
            # Processing phase: Copy transcript to clipboard
            try:
                # Use pyclip to put transcript in clipboard
                import pyclip
                pyclip.copy(transcript)
                print(f"[INFO] Transcript copied to clipboard")
            except Exception as e:
                print(f"[WARNING] Failed to copy to clipboard: {e}")
            
            # Teletype already playing from immediate feedback - no transition needed
            
            # Start teletype sound immediately before typing (MP3 has built-in 0.2s silence for sync)
            print("[INFO] Starting teletype sound with built-in timing...")
            asyncio.create_task(self._play_claude_code_confirmation())
            
            # Delay to ensure audio is playing before subprocess blocks
            await asyncio.sleep(0.45)
            
            # Run the voice injector with sudo to create virtual keyboard
            print("[INFO] Creating virtual keyboard for injection...")
            result = subprocess.run([
                'sudo', venv_python, injector_script, '--inject-text', transcript
            ], capture_output=True, text=True, timeout=30)
            
            # Stop teletype sound when injection completes
            print("[DEBUG] Injection complete, stopping typing phase sound...")
            await self.audio_coordinator.play_phase_sound("complete")
            
            if result.returncode == 0:
                print("[INFO] Successfully injected transcript to Claude Code")
            else:
                print(f"[ERROR] Injection failed: {result.stderr}")
                
        except Exception as e:
            print(f"[ERROR] Failed to inject to Claude Code: {e}")
            import traceback
            traceback.print_exc()
    
    async def send_enter_key(self):
        """Send Enter key using virtual keyboard for 'send now' wake word"""
        import subprocess
        import os
        
        try:
            print("[INFO] Sending Enter key to focused application")
            
            # Path to our Enter key sender
            project_root = os.path.dirname(os.path.abspath(__file__))
            enter_script = os.path.join(project_root, "claude", "send_enter.py")
            venv_python = os.path.join(project_root, "venv", "bin", "python")
            
            # Run the Enter key sender with sudo
            result = subprocess.run([
                'sudo', venv_python, enter_script
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("[INFO] Successfully sent Enter key")
            else:
                print(f"[ERROR] Failed to send Enter key: {result.stderr}")
                
        except Exception as e:
            print(f"[ERROR] Failed to send Enter key: {e}")
            import traceback
            traceback.print_exc()
    
    def _should_route_to_claude_code_from_wake(self, wake_event_source: str) -> bool:
        """Check if wake event is for Claude Code (for confirmation sound)"""
        return self._should_route_to_claude_code(wake_event_source) or wake_event_source == "keyboard_code"
    
    async def _play_claude_code_confirmation(self):
        """Play teletype sound with built-in timing (MP3 has 0.2s silence at start)"""
        teletype = str(SOUND_BASE_PATH / "sound_effects" / "teletype.mp3")
        print("[DEBUG] Playing teletype sound with built-in sync timing!")
        if os.path.exists(teletype):
            try:
                await self.audio_coordinator.play_phase_sound("processing", teletype)
                print("[DEBUG] Teletype sound playback completed")
            except Exception as e:
                print(f"[DEBUG] Teletype playback failed: {e}")
                # Try direct fallback
                try:
                    await self.audio_coordinator.play_audio_file(teletype)
                    print("[DEBUG] Teletype fallback playback completed")
                except Exception as e2:
                    print(f"[DEBUG] Teletype fallback also failed: {e2}")
        else:
            print(f"[DEBUG] Teletype file not found: {teletype}")
    
    async def _delayed_teletype_sound(self, teletype_file):
        """Play teletype sound after 0.6s delay"""
        await asyncio.sleep(0.6)
        print("[DEBUG] Starting teletype sound after delay...")
        try:
            await self.audio_coordinator.play_phase_sound("processing", teletype_file)
            print("[DEBUG] Teletype sound playback completed")
        except Exception as e:
            print(f"[DEBUG] Teletype sound playback failed: {e}")
            # Try direct fallback
            try:
                await self.audio_coordinator.play_audio_file(teletype_file)
                print("[DEBUG] Teletype fallback playback completed")
            except Exception as e2:
                print(f"[DEBUG] Teletype fallback also failed: {e2}")
    
    def _switch_to_persona(self, persona: str):
        """
        Switch both display profile and TTS voice configuration
        
        Args:
            persona: 'laura' for LAURA persona, 'claude_code' for Claude Code persona
        """
        import json
        
        print(f"[INFO] Switching to {persona} persona")
        
        # Switch display profile
        if persona == 'laura':
            self.state_tracker.display_profile = 'laura'
        elif persona == 'claude_code':
            self.state_tracker.display_profile = 'claude_code'
        
        # Update TTS voice configuration
        try:
            # Update TTS server voices.json
            voices_config_path = "/home/user/rp_client/TTS/config/voices.json"
            with open(voices_config_path, 'r') as f:
                voices_config = json.load(f)
            
            if persona == 'laura':
                voices_config['active_voice'] = 'qEwI395unGwWV1dn3Y65'  # LAURA voice
            elif persona == 'claude_code':
                voices_config['active_voice'] = 'uY96J30mUhYUIymmD5cu'  # Claude Code voice
            
            with open(voices_config_path, 'w') as f:
                json.dump(voices_config, f, indent=2)
            
            # Notify TTS server to reload configuration
            import aiohttp
            import asyncio
            
            async def reload_tts_config():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post('http://localhost:5001/reload_config') as response:
                            if response.status == 200:
                                print(f"[INFO] TTS server reloaded config for {persona} persona")
                            else:
                                print(f"[WARNING] TTS reload returned status {response.status}")
                except Exception as e:
                    print(f"[WARNING] Could not notify TTS server to reload: {e}")
            
            # Run the reload in the background
            asyncio.create_task(reload_tts_config())
            
            print(f"[INFO] Updated TTS voice to {persona} persona")
            
        except Exception as e:
            print(f"[ERROR] Failed to update TTS voice configuration: {e}")
    
    def _should_route_to_claude_code(self, wake_event_source: str) -> bool:
        """
        Determine if wake event should route to Claude Code CLI
        
        Args:
            wake_event_source: The wake event source (e.g., "wakeword (YourNewModel.pmdl)")
            
        Returns:
            bool: True if should route to Claude Code, False for regular chat
        """
        if not wake_event_source or 'wakeword' not in wake_event_source:
            return False
            
        # Extract model name from wake event source
        if '(' in wake_event_source and ')' in wake_event_source:
            model_name = wake_event_source.split('(')[1].rstrip(')')
            
            # Define wake words that should route to Claude Code
            claude_code_wake_words = [
                "claudecode.pmdl",
            ]
            
            return model_name in claude_code_wake_words
            
        return False
    
    def _should_send_enter_key(self, wake_event_source: str) -> bool:
        """
        Determine if wake event should send Enter key
        
        Args:
            wake_event_source: The wake event source (e.g., "wakeword (send_now.pmdl)")
            
        Returns:
            bool: True if should send Enter key, False otherwise
        """
        if not wake_event_source or 'wakeword' not in wake_event_source:
            return False
            
        # Extract model name from wake event source
        if '(' in wake_event_source and ')' in wake_event_source:
            model_name = wake_event_source.split('(')[1].rstrip(')')
            
            # Define wake words that should send Enter
            send_enter_wake_words = [
                "send_now.pmdl",
            ]
            
            return model_name in send_enter_wake_words
            
        return False
    
    def _should_send_note_to_mac(self, wake_event_source: str) -> bool:
        """
        Determine if wake event should trigger note transfer to Mac
        
        Args:
            wake_event_source: The wake event source (e.g., "wakeword (sendnote.pmdl)")
            
        Returns:
            bool: True if should send note to Mac, False otherwise
        """
        if not wake_event_source or 'wakeword' not in wake_event_source:
            return False
            
        # Extract model name from wake event source
        if '(' in wake_event_source and ')' in wake_event_source:
            model_name = wake_event_source.split('(')[1].rstrip(')')
            
            # Define wake words that should trigger note transfer
            note_transfer_wake_words = [
                "sendnote.pmdl",
            ]
            
            return model_name in note_transfer_wake_words
            
        return False
    
    async def send_note_to_mac(self):
        """Send pi500_note.txt to Mac server via MCP endpoint"""
        try:
            from send_note_to_mac import Pi500NoteSender
            
            sender = Pi500NoteSender()
            result = sender.send_note()
            
            if result["success"]:
                # Success - play confirmation sound and speak result
                success_sound = str(SOUND_BASE_PATH / "sound_effects" / "successfulloadup.mp3")
                if os.path.exists(success_sound):
                    await self.audio_coordinator.play_audio_file(success_sound)
                
                message = "Note successfully sent to Mac server!"
                print(f"[INFO] {message}")
                
                # Speak confirmation
                await self.tts_handler.speak_text(
                    message,
                    voice_params={"persona": "laura"},
                    coordinator=self.audio_coordinator
                )
                
            else:
                # Error - play error sound and speak error
                error_sound = str(SOUND_BASE_PATH / "sound_effects" / "error.mp3")
                if os.path.exists(error_sound):
                    await self.audio_coordinator.play_audio_file(error_sound)
                
                message = f"Failed to send note: {result.get('error', 'Unknown error')}"
                print(f"[ERROR] {message}")
                
                # Speak error
                await self.tts_handler.speak_text(
                    "Sorry, I couldn't send the note to Mac. Check the connection.",
                    voice_params={"persona": "laura"},
                    coordinator=self.audio_coordinator
                )
                
        except Exception as e:
            print(f"[ERROR] Exception in send_note_to_mac: {e}")
            
            # Play error sound
            error_sound = str(SOUND_BASE_PATH / "sound_effects" / "error.mp3")
            if os.path.exists(error_sound):
                await self.audio_coordinator.play_audio_file(error_sound)
            
            # Speak error
            await self.tts_handler.speak_text(
                "Sorry, there was an error sending the note.",
                voice_params={"persona": "laura"},
                coordinator=self.audio_coordinator
            )


async def main():
    """Main entry point with multi-task architecture"""
    print("=== STARTING MAIN ===", flush=True)
    from config.client_config import load_client_settings
    print("=== LOADING SETTINGS ===", flush=True)
    load_client_settings()
    print("=== SETTINGS LOADED ===", flush=True)
    
    # Check VOSK readiness
    print("[INFO] Checking VOSK server readiness...")
    vosk_ready = await ensure_vosk_ready(timeout=10)
    if vosk_ready:
        print("[INFO] VOSK server is ready")
    else:
        print("[WARNING] VOSK server not ready - speech features will be limited")

    print("[INFO] Initializing Pi MCP Client...")
    client = PiMCPClient(server_url=SERVER_URL, device_id=DEVICE_ID)
    print("[INFO] Client initialized successfully")
    
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received.")
    finally:
        print("[INFO] Main function finished. Performing final cleanup...")
        await client.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Application terminated by user.")
    finally:
        print("[INFO] Application shutdown complete.")
