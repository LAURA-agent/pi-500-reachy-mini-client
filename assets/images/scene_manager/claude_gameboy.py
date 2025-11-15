#!/usr/bin/env python3
"""
Gameboy Receiver - Claude SVG Animator (Fixed Version)
Pre-analyzes audio for smooth animation synchronized with playback.
Designed for CM4 with 640x480 display.
"""

import asyncio
import json
import os
import sys

# Set DISPLAY for pygame when running via SSH
if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'
import logging
import time
import math
import io
import threading
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Animation & Audio Libraries
import pygame
import cairosvg
import numpy as np
import pyaudio
from pydub import AudioSegment
import httpx
import websockets

# Import API keys from config
import sys
sys.path.insert(0, '/home/user/gpi-claude-assistant/config')
try:
    from client_secret import ELEVENLABS_API_KEY
except ImportError:
    print("ERROR: Could not import ELEVENLABS_API_KEY from client_secret.py")
    print("Make sure /home/user/gpi-claude-assistant/config/client_secret.py exists")
    sys.exit(1)

ELEVENLABS_VOICE_ID = 'uY96J30mUhYUIymmD5cu'
HUB_URL = 'ws://174.165.47.128:8766/ws'

# Display Settings
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
STAR_BASE_SIZE = 240  # Base size for the star (reduced for more room to bounce)

# Animation Settings
DEFAULT_COLOR = "#c04015"  # Changed from #ca7633 to use the former arrogant orange
SNARKY_COLOR = "#b8230e"  # Deeper, redder version of arrogant orange
TARGET_FPS = 24
FRAME_TIME = 1.0 / TARGET_FPS

# Audio Analysis Settings
RMS_DIVISOR = 13700.0  # Based on actual Claude voice testing
CHUNK_SIZE = 1024
SAMPLE_RATE = 44100  # Standard sample rate
CHUNK_DURATION = CHUNK_SIZE / SAMPLE_RATE  # Duration of each chunk in seconds
SCALE_MULTIPLIER = 0.6  # 60% size variation (0.4 to 1.6 range) - much more dramatic speaking animation

# Claude SVG
CLAUDE_SVG_XML = """
<?xml version="1.0" encoding="UTF-8"?>
<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1104.33 1080">
  <defs><style>.cls-1{fill:FILL_COLOR_PLACEHOLDER;fill-rule:evenodd;stroke-width:0px;}</style></defs>
  <path class="cls-1" d="m290.56,0c81.75.46,84.09,61.27,107.89,117.66,30.05,71.18,72.18,137.58,106.18,205.47,10.09,20.14,21.33,71.09,36.82,83.55l3.43-.85c16.5-16.85,5.12-57.54,9.42-85.26,5.71-35.52,11.42-71.05,17.13-106.57,9.42-69.87-12.17-188.73,60.8-195.24,101.64,2.86-26.39,304.8-14.56,353.82-1.05,1.06,9.19,18.29,19.69,26.43,49.18-12.79,67.76-83.55,83.06-127.03,29.77-39.15,57.83-78.33,92.48-112.54,62.64-78.69,144.9-29.88,106.18,63.95-47.66,53.76-86.78,114.27-131.01,170.51-55.81,95.17-118.49,112.58,23.97,73.32,61.22-17.12,133.51-15.67,191.81-39.21,49.14-6.7,106.58,25.9,47.95,70.76-13.98,1.7-26.82,3.05-41.96,5.11-12.63,4.9-24.32,12.41-37.68,15.35s-41.49,5.37-76.21,9.38c-42.94,9.77-61.54,18.31-85.63,23.02-19.81,3.87-77.62,6.73-86.48,18.76,14.78,13.57,33.84,11.88,59.08,15.35,47.09,7.39,118.33,5.75,166.12,11.94,36.42,4.71,64.74-2.09,88.2,12.79,15.52,9.84,49.33,21.35,39.39,54.57-6.27,20.94-44.55,35.1-75.35,28.99-15.29-3.03-35.06-13.64-47.95-17.05-54.26-6.88-110.68-15.42-160.12-34.96-10.86-4.41-79.06-18.69-82.2-10.23-6.07,16.32,39.32,42.52,47.95,51.16,50.93,50.9,108.88,95.26,158.41,147.5,12.88,13.59,66.91,51.22,34.25,77.59-32.02.18-49.84-26.75-69.36-40.92-57.19-41.54-133.79-110.08-183.24-159.43-4,1.42-2.85,1.99-6.85,3.41-4.93,20.95,118.75,174.1,131.87,206.33,11.59,28.47,19.41,73.37-15.41,81.85-34.43,8.38-48.28-26.65-59.94-43.48-35.66-51.46-74.37-103.99-109.6-156.88-33.98-75.56-60.52-101.17-61.65,5.97-7.26,52.15-12.32,120.12-19.69,168.81-2,26.71-4,53.43-5.99,80.14-3.51,9.02-34.81,30.08-49.66,25.58-41.57-13.35-29.92-90.13-13.7-125.33,2.28-19.89,4.56-39.79,6.85-59.68,4.28-13.07,8.56-26.15,12.84-39.22,5.15-24.24,6-53.05,11.99-75.03,5.64-20.69,21.58-83.25.86-86.96-16.15,11.47-21.49,34.91-33.39,51.16-21.82,29.77-49.13,58.89-71.07,88.67-24.87,28.79-143.72,224.51-178.96,143.24-7.12-25.21,47.55-80.62,59.94-96.35,41.18-52.3,78.94-103.95,119.02-157.73,11.12-14.92,49.77-42.17,53.09-59.68l-3.43-5.12c-24.19-5.3-44.57,25.75-63.36,34.1-42.47,18.89-76.18,54.51-114.74,76.73-28.5,16.42-56.11,28.97-81.35,48.6-96.97,61.75-149.24-14.44-38.53-65.64,21.79-16.35,44.68-34.88,68.5-49.45,19.9-12.18,40.46-16.43,59.94-28.99,16.36-10.54,38.24-25.49,56.51-33.25,21.29-9.04,83.12-34.99,71.93-53.71-11.19-18.72-72.04-3.45-107.03-7.67-77.26-9.32-154.96-1.62-223.49-12.79-38.06-6.2-103.93,8.54-93.33-45.19,7.06-35.78,82.33-9.52,113.88-11.94,21.74-1.67,38.34,4.1,56.51,6.82,58.26,5.54,107.77,5.82,164.4,7.67,24.52,3.31,61.72,17.51,82.2,8.53,9.7-26.43-65.76-61.85-85.63-73.32-63.41-36.63-123.31-87.18-183.24-127.89-15.26-10.37-45.77-21.74-55.66-35.81-12.05-17.14-16.1-54.06-5.14-71.62,28.63-41.92,89.46.98,113.88,23.87,35.05,24.15,71.65,46.75,101.04,78.44,16.08,11.54,33.83,18.26,48.81,30.69,11.05,9.17,48.45,49.86,60.8,44.33,7.74-25.08-21.9-52.81-29.97-72.47-28.94-70.56-69.1-126.28-104.47-191.83-31.94-44.8-54.79-97.69,11.99-135.56Z"/>
</svg>
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DialogueEntry:
    turn: int
    text: str
    mood: str = "neutral"
    audio_data: Optional[bytes] = None
    audio_ready: asyncio.Event = field(default_factory=asyncio.Event)
    amplitude_timeline: List[float] = field(default_factory=list)
    duration: float = 0.0
    played: bool = False
    tts_failed: bool = False

class SvgAnimator:
    """Manages SVG rendering and animation with pre-computed amplitude data."""

    def __init__(self):
        self.initialized = False
        self.screen = None
        self.master_sprites: Dict[str, pygame.Surface] = {}
        self.current_master_sprite: Optional[pygame.Surface] = None

        # Animation state
        self.is_speaking = False
        self.current_scale = 1.0
        self.target_scale = 1.0
        self.rotation_angle = 0.0
        self.playback_start_time: Optional[float] = None
        self.amplitude_timeline: List[float] = []
        self.audio_duration = 0.0

        # Reference to audio playback state (will be set by GameboyReceiver)
        self.audio_playback_state = None

        # Bouncing motion state (DVD screensaver style)
        self.x = DISPLAY_WIDTH // 2  # Start in center
        self.y = DISPLAY_HEIGHT // 2
        self.vx = 2.5  # Initial velocity (pixels per frame)
        self.vy = 1.8  # Slightly different to avoid patterns
        self.corner_threshold = 30  # Pixels from corner to trigger avoidance
        self.last_corner_time = 0  # Track when we last avoided a corner
        self.corner_cooldown = 2.0  # Seconds between corner avoidance adjustments

        try:
            # Initialize pygame but disable mixer to avoid audio conflicts with pyaudio
            pygame.mixer.quit()  # Ensure mixer is not running
            pygame.init()
            pygame.mixer.quit()  # Disable mixer after init to prevent conflicts
            self.screen = pygame.display.set_mode(
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                pygame.NOFRAME
            )
            pygame.display.set_caption("Gameboy - Claude")

            # Render SVG for each mood at appropriate size
            logger.info("Rendering SVG assets...")
            for mood, color in [('default', DEFAULT_COLOR), ('snarky', SNARKY_COLOR)]:
                svg_data = CLAUDE_SVG_XML.strip().replace("FILL_COLOR_PLACEHOLDER", color)
                png_data = cairosvg.svg2png(
                    bytestring=svg_data.encode('utf-8'),
                    output_width=STAR_BASE_SIZE
                )
                image_bytes = io.BytesIO(png_data)
                self.master_sprites[mood] = pygame.image.load(image_bytes).convert_alpha()

            self.current_master_sprite = self.master_sprites['default']
            self.initialized = True
            logger.info(f"✅ SVG Animator initialized (base size: {STAR_BASE_SIZE}px)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize SVG Animator: {e}", exc_info=True)
            self.initialized = False

    def set_speaking_state(self, is_speaking: bool, amplitude_timeline: List[float] = None, duration: float = 0):
        """Start or stop speaking animation with pre-computed amplitude data."""
        self.is_speaking = is_speaking
        if is_speaking and amplitude_timeline:
            self.playback_start_time = time.time()
            self.amplitude_timeline = amplitude_timeline
            self.audio_duration = duration
            logger.info(f"📊 Started animation with {len(amplitude_timeline)} amplitude points")
        else:
            self.playback_start_time = None
            self.amplitude_timeline = []
            self.audio_duration = 0.0

    def set_mood(self, mood: str):
        """Change the SVG color based on mood."""
        mood_key = 'snarky' if mood == 'snarky' else 'default'
        if mood_key in self.master_sprites:
            self.current_master_sprite = self.master_sprites[mood_key]
            logger.info(f"🎨 Mood set to: {mood_key}")

    def update_animation(self):
        """Update animation frame using pre-computed amplitude data."""
        if not self.initialized or not self.current_master_sprite:
            return

        # Calculate current animation parameters
        elapsed = 0.0

        # Use audio stream clock if available
        if self.is_speaking and self.audio_playback_state and self.amplitude_timeline:
            if self.audio_playback_state.get("is_active"):
                stream = self.audio_playback_state.get("stream")
                if stream:
                    try:
                        # Get elapsed time from audio stream clock
                        current_time = stream.get_time()
                        start_time = self.audio_playback_state.get("start_time", 0)
                        elapsed = current_time - start_time
                    except Exception as e:
                        # Fallback to frames written if get_time() fails
                        frames_written = self.audio_playback_state.get("frames_written", 0)
                        sample_rate = self.audio_playback_state.get("sample_rate", 44100)
                        if sample_rate > 0:
                            elapsed = frames_written / sample_rate

                if elapsed > 0 and elapsed < self.audio_duration:
                    # Find the right amplitude value for current time
                    timeline_index = int(elapsed / CHUNK_DURATION)
                    if timeline_index < len(self.amplitude_timeline):
                        amplitude = self.amplitude_timeline[timeline_index]
                        self.target_scale = 1.0 + (amplitude * SCALE_MULTIPLIER)
                    else:
                        self.target_scale = 1.0
                else:
                    # Audio finished or not started
                    self.target_scale = 1.0

                self.rotation_angle += 1.8  # Faster rotation when talking
            else:
                # Audio not active, use idle animation
                breathing = (math.sin(time.time() * 2) + 1) / 2
                self.target_scale = 0.97 + (breathing * 0.06)  # 97% to 103%
                self.rotation_angle += 0.5  # Slow rotation when idle
        else:
            # Idle animation - gentle breathing effect
            breathing = (math.sin(time.time() * 2) + 1) / 2
            self.target_scale = 0.97 + (breathing * 0.06)  # 97% to 103%
            self.rotation_angle += 0.5  # Slow rotation when idle

        # Smooth scale transitions
        self.current_scale += (self.target_scale - self.current_scale) * 0.25

        # Update bouncing position
        self.update_bouncing_position()

        # Apply transformations
        scaled_size = int(STAR_BASE_SIZE * self.current_scale)
        scaled_sprite = pygame.transform.smoothscale(
            self.current_master_sprite,
            (scaled_size, scaled_size)
        )

        # Rotate the scaled sprite
        rotated_sprite = pygame.transform.rotate(scaled_sprite, self.rotation_angle)

        # Position at current bouncing location
        rect = rotated_sprite.get_rect(center=(int(self.x), int(self.y)))

        # Draw to screen
        self.screen.fill((0, 0, 0))  # Black background
        self.screen.blit(rotated_sprite, rect.topleft)
        pygame.display.flip()

    def update_bouncing_position(self):
        """Update position with DVD-style bouncing and corner avoidance."""
        # Calculate next position
        next_x = self.x + self.vx
        next_y = self.y + self.vy

        # Get sprite bounds (accounting for current scale and rotation)
        sprite_radius = (STAR_BASE_SIZE * self.current_scale) // 2

        # Check for wall collisions and bounce
        hit_wall = False

        # Left/right walls
        if next_x - sprite_radius <= 0:
            self.vx = abs(self.vx)  # Bounce right
            next_x = sprite_radius
            hit_wall = True
        elif next_x + sprite_radius >= DISPLAY_WIDTH:
            self.vx = -abs(self.vx)  # Bounce left
            next_x = DISPLAY_WIDTH - sprite_radius
            hit_wall = True

        # Top/bottom walls
        if next_y - sprite_radius <= 0:
            self.vy = abs(self.vy)  # Bounce down
            next_y = sprite_radius
            hit_wall = True
        elif next_y + sprite_radius >= DISPLAY_HEIGHT:
            self.vy = -abs(self.vy)  # Bounce up
            next_y = DISPLAY_HEIGHT - sprite_radius
            hit_wall = True

        # Corner avoidance logic - the infuriating part!
        if hit_wall:
            current_time = time.time()
            # Check if we're dangerously close to a corner
            dist_to_corners = [
                math.sqrt((next_x - 0)**2 + (next_y - 0)**2),  # Top-left
                math.sqrt((next_x - DISPLAY_WIDTH)**2 + (next_y - 0)**2),  # Top-right
                math.sqrt((next_x - 0)**2 + (next_y - DISPLAY_HEIGHT)**2),  # Bottom-left
                math.sqrt((next_x - DISPLAY_WIDTH)**2 + (next_y - DISPLAY_HEIGHT)**2)  # Bottom-right
            ]

            min_corner_dist = min(dist_to_corners)

            # If we're about to hit a corner (or get really close), adjust trajectory
            if min_corner_dist < sprite_radius + self.corner_threshold:
                if current_time - self.last_corner_time > self.corner_cooldown:
                    # Add a small deflection to avoid the perfect corner hit
                    # The amount is calculated to be just enough to miss
                    deflection = 0.15 + (math.sin(current_time * 3.7) * 0.1)  # Slight randomness

                    # Adjust velocity vectors slightly
                    if abs(self.vx) > abs(self.vy):
                        self.vy *= (1 + deflection)
                    else:
                        self.vx *= (1 + deflection)

                    # Normalize to maintain roughly same speed
                    speed = math.sqrt(self.vx**2 + self.vy**2)
                    target_speed = 3.2  # Desired speed
                    if speed > 0:
                        self.vx = (self.vx / speed) * target_speed
                        self.vy = (self.vy / speed) * target_speed

                    self.last_corner_time = current_time
                    logger.debug(f"Corner avoidance triggered! Distance: {min_corner_dist:.1f}px")

        # Update position
        self.x = next_x
        self.y = next_y

    def cleanup(self):
        """Clean up pygame resources."""
        if self.initialized:
            pygame.quit()
            logger.info("🧹 SVG Animator cleaned up")

class GameboyReceiver:
    def __init__(self):
        self.speaker_identity = "claude"
        self.dialogue_queue: Dict[int, DialogueEntry] = {}
        self.animator = SvgAnimator()
        self.play_next_turn_event = asyncio.Event()
        self.playback_lock = asyncio.Lock()
        self.background_tasks = set()
        self.websocket: Optional[websockets.client.WebSocketClientProtocol] = None
        self.audio_thread: Optional[threading.Thread] = None
        self.stop_audio = threading.Event()
        self.is_playing_audio = False
        self.pending_turn = None  # Track which turn is queued to play

        # Shared audio playback state for sync
        self.audio_playback_state = {
            "stream": None,
            "is_active": False,
            "start_time": 0.0,
            "frames_written": 0,
            "sample_rate": 0
        }

        logger.info(f"🚀 Gameboy Receiver initialized for '{self.speaker_identity}'")

    def _handle_task_completion(self, task):
        """Handle background task completion and log exceptions."""
        self.background_tasks.discard(task)
        if task.cancelled():
            logger.info("Background task was cancelled")
        elif task.exception():
            logger.error(f"❌ Background task failed with exception: {task.exception()}", exc_info=task.exception())
        else:
            logger.info("Background task completed successfully")

    async def pre_analyze_audio(self, audio_data: bytes) -> Tuple[List[float], float]:
        """Pre-analyze audio to create amplitude timeline."""
        try:
            # Load audio
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            duration = len(audio_segment) / 1000.0  # Duration in seconds

            # Process in chunks to build amplitude timeline
            raw_data = audio_segment.raw_data
            amplitude_timeline = []

            for i in range(0, len(raw_data), CHUNK_SIZE):
                chunk = raw_data[i:i + CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    break  # Skip incomplete final chunk

                # Calculate RMS for this chunk
                audio_np = np.frombuffer(chunk, dtype=np.int16)
                if audio_np.size > 0:
                    rms = np.sqrt(np.mean(audio_np.astype(float)**2))
                    normalized = min(rms / RMS_DIVISOR, 1.0)
                    amplitude_timeline.append(normalized)
                else:
                    amplitude_timeline.append(0.0)

            logger.info(f"📊 Pre-analyzed audio: {duration:.1f}s, {len(amplitude_timeline)} chunks")
            return amplitude_timeline, duration

        except Exception as e:
            logger.error(f"❌ Failed to pre-analyze audio: {e}")
            return [], 0.0

    async def generate_tts(self, entry: DialogueEntry):
        """Generate TTS and pre-analyze the audio."""
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            data = {
                "text": entry.text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=20.0)
                response.raise_for_status()
                entry.audio_data = response.content

                # Pre-analyze the audio for animation
                entry.amplitude_timeline, entry.duration = await self.pre_analyze_audio(
                    entry.audio_data
                )

                logger.info(f"✅ TTS generated and analyzed for turn {entry.turn}")

        except Exception as e:
            logger.error(f"❌ TTS FAILED for turn {entry.turn}: {e}")
            entry.tts_failed = True
        finally:
            entry.audio_ready.set()

    def play_audio_thread(self, audio_data: bytes):
        """Play audio in a separate thread (non-blocking)."""
        p = pyaudio.PyAudio()
        stream = None
        self.is_playing_audio = True

        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")

            # Open audio stream with larger buffer to prevent underruns
            # Force use of USB Audio Device (card 0) to avoid ALSA errors
            stream = p.open(
                format=p.get_format_from_width(audio_segment.sample_width),
                channels=audio_segment.channels,
                rate=audio_segment.frame_rate,
                output=True,
                output_device_index=0,  # Force USB Audio Device
                frames_per_buffer=2048  # Larger buffer for stability
            )

            # Store stream in shared state for animation sync
            self.audio_playback_state["stream"] = stream
            self.audio_playback_state["is_active"] = True
            self.audio_playback_state["start_time"] = stream.get_time()
            self.audio_playback_state["frames_written"] = 0
            self.audio_playback_state["sample_rate"] = audio_segment.frame_rate

            # Play audio in chunks
            chunk_size = 2048  # Match buffer size
            frames_per_chunk = chunk_size // (audio_segment.channels * audio_segment.sample_width)
            for i in range(0, len(audio_segment.raw_data), chunk_size):
                if self.stop_audio.is_set():
                    break
                chunk = audio_segment.raw_data[i:i + chunk_size]
                # Non-blocking write with timeout to prevent hanging
                try:
                    stream.write(chunk, exception_on_underflow=False)
                    self.audio_playback_state["frames_written"] += frames_per_chunk
                except Exception as write_error:
                    logger.warning(f"Audio write error: {write_error}")
                    break

            # Signal that audio playback is complete (before cleanup)
            if hasattr(self, 'audio_completion_event'):
                self.audio_completion_event.set()

        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            if hasattr(self, 'audio_completion_event'):
                self.audio_completion_event.set()
        finally:
            # Clean up shared state - critical for preventing stuck animations
            self.audio_playback_state["is_active"] = False
            self.audio_playback_state["stream"] = None
            self.audio_playback_state["frames_written"] = 0
            self.audio_playback_state["sample_rate"] = 0

            self.is_playing_audio = False
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    async def send_completion_signal(self, turn: int):
        """Send playback completion signal to hub."""
        if self.websocket and self.websocket.state == websockets.protocol.State.OPEN:
            try:
                await self.websocket.send(json.dumps({
                    "type": "playback_complete",
                    "from_device": self.speaker_identity,
                    "turn": turn
                }))
                logger.info(f"✅ Completion signal sent for turn {turn}")
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Could not send completion: WebSocket closed")
        else:
            logger.warning("Could not send completion: WebSocket not connected")

    def add_dialogue(self, turn: int, text: str, is_first: bool, mood: str = "neutral"):
        """Add dialogue to queue and start TTS generation."""
        if turn in self.dialogue_queue:
            logger.warning(f"Duplicate dialogue for turn {turn}, ignoring")
            return

        if not self.dialogue_queue:
            self.current_turn_to_play = turn
            logger.info(f"First turn: {turn}")

        entry = DialogueEntry(turn=turn, text=text, mood=mood)
        self.dialogue_queue[turn] = entry
        logger.info(f"📥 Added dialogue for turn {turn} (mood: {mood})")

        # Start TTS generation and pre-analysis
        task = asyncio.create_task(self.generate_tts(entry))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        if is_first or turn == self.current_turn_to_play:
            self.play_next_turn_event.set()

    async def playback_manager(self):
        """Manage audio playback with animation."""
        logger.info("▶️ Playback Manager started")

        # Connect animator to audio playback state
        self.animator.audio_playback_state = self.audio_playback_state

        while True:
            await self.play_next_turn_event.wait()
            self.play_next_turn_event.clear()

            async with self.playback_lock:
                entry = self.dialogue_queue.get(self.current_turn_to_play)
                if not entry:
                    continue

                await entry.audio_ready.wait()

                if entry.played:
                    continue

                if entry.tts_failed:
                    logger.error(f"🚫 Skipping turn {entry.turn} due to TTS failure")
                elif entry.audio_data:
                    logger.info(f"🔊 Playing turn {entry.turn}")

                    # Set mood and start animation with pre-computed data
                    self.animator.set_mood(entry.mood)
                    self.animator.set_speaking_state(
                        True,
                        entry.amplitude_timeline,
                        entry.duration
                    )

                    # Start audio playback in thread
                    self.stop_audio.clear()
                    self.audio_completion_event = threading.Event()
                    self.audio_thread = threading.Thread(
                        target=self.play_audio_thread,
                        args=(entry.audio_data,)
                    )
                    self.audio_thread.start()

                    # Wait for audio playback to complete (not thread cleanup)
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.audio_completion_event.wait
                    )

                    # Send completion signal immediately after audio stops
                    entry.played = True
                    await self.send_completion_signal(entry.turn)

                    # Stop animation (can happen after completion signal)
                    self.animator.set_speaking_state(False)

                    # Clean up thread in background (don't wait for it)
                    if self.audio_thread:
                        threading.Thread(target=lambda: self.audio_thread.join()).start()

    async def websocket_handler(self):
        """Handle WebSocket connection to hub."""
        while True:
            try:
                async with websockets.connect(HUB_URL) as websocket:
                    self.websocket = websocket
                    logger.info(f"🔗 Connected to Hub at {HUB_URL}")

                    # Register as claude device
                    await websocket.send(json.dumps({
                        "type": "device_registration",
                        "speaker": self.speaker_identity
                    }))

                    async for message in websocket:
                        data = json.loads(message)
                        msg_type = data.get('type')

                        if msg_type == 'dialogue':
                            logger.info(f"✉️ Received dialogue for turn {data.get('turn')}")
                            self.add_dialogue(
                                turn=data.get('turn'),
                                text=data.get('text'),
                                is_first=data.get('is_first', False),
                                mood=data.get('mood', 'neutral')
                            )

                        elif msg_type == 'playback_complete':
                            turn = data.get('turn')
                            logger.info(f"✉️ Completion signal for turn {turn}")
                            self.current_turn_to_play = turn + 1
                            self.play_next_turn_event.set()

                        elif msg_type == 'conversation_reset':
                            logger.info("🔄 Conversation reset")
                            self.dialogue_queue.clear()
                            self.current_turn_to_play = 1
                            self.play_next_turn_event.clear()

            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket = None
                logger.info("Retrying connection in 5 seconds...")
                await asyncio.sleep(5)

    async def animation_loop(self):
        """Run animation updates at target FPS."""
        while True:
            if self.animator.initialized:
                self.animator.update_animation()

                # Process ALL events to prevent queue buildup
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        await self.stop()
                        return
                    # Consume all other events including MOUSEMOTION, MOUSEBUTTONDOWN, etc.
                    # This prevents event queue from filling up and blocking

            # Always yield to other tasks after animation update
            await asyncio.sleep(0)
            await asyncio.sleep(FRAME_TIME)

    async def start(self):
        """Start all background tasks."""
        if self.animator.initialized:
            tasks = [
                self.websocket_handler(),
                self.playback_manager(),
                self.animation_loop()
            ]
            for task_coro in tasks:
                task = asyncio.create_task(task_coro)
                self.background_tasks.add(task)
                task.add_done_callback(lambda t: self._handle_task_completion(t))

            logger.info("✅ Receiver started")
        else:
            logger.error("❌ Cannot start: animator failed to initialize")
            await self.stop()

    async def stop(self):
        """Gracefully shutdown."""
        logger.info("🛑 Shutting down...")

        # Stop audio playback
        self.stop_audio.set()
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)

        # Close websocket
        if self.websocket and self.websocket.state == websockets.protocol.State.OPEN:
            await self.websocket.close()

        # Cancel tasks
        for task in self.background_tasks:
            task.cancel()
        await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Clean up animator
        self.animator.cleanup()

        logger.info("Goodbye")
        sys.exit(0)

async def main():
    receiver = GameboyReceiver()
    try:
        await receiver.start()
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown signal received")
    finally:
        await receiver.stop()

if __name__ == "__main__":
    asyncio.run(main())