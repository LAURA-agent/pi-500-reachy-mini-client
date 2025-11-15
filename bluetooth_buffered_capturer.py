#!/usr/bin/env python3
"""Buffered Bluetooth audio capture with sentence-based analysis and synchronized playback.

Uses application-level rerouting:
1. Detect bluez_input stream
2. Mute it (silent to user)
3. Capture from monitor (receives muted stream)
4. Buffer until sentence boundary (pause detection)
5. Analyze + playback with synchronized motion
"""

import asyncio
import subprocess
import struct
import numpy as np
import tempfile
import wave
import time
import os
from typing import Optional
from pathlib import Path


class BluetoothBufferedCapturer:
    """Capture Bluetooth audio silently, buffer by sentence, then play with motion sync."""

    def __init__(self, audio_coordinator, movement_manager, state_tracker):
        """Initialize buffered capturer.

        Args:
            audio_coordinator: AudioCoordinator instance for TTS playback and analysis
            movement_manager: MovementManager instance for motion control
            state_tracker: StateTracker instance for state management
        """
        self.audio_coordinator = audio_coordinator
        self.movement_manager = movement_manager
        self.state_tracker = state_tracker

        # Stream detection
        self.bluetooth_stream_id = None
        self.stream_name_pattern = "bluez_input"

        # Audio capture
        self.capture_process = None
        self.capture_task = None
        self.is_capturing = False

        # Buffering parameters
        self.sample_rate = 16000
        self.channels = 2
        self.chunk_size = 1024  # samples per channel
        self.bytes_per_sample = 2  # s16le
        self.bytes_per_frame = self.bytes_per_sample * self.channels

        # Sentence detection
        self.audio_buffer = []  # List of audio chunks (bytes)
        self.silence_threshold = 0.02  # Energy threshold for silence
        self.silence_duration = 0.8  # Seconds of silence = sentence boundary
        self.min_sentence_duration = 0.5  # Minimum sentence length
        self.silence_frames = 0
        self.silence_frames_needed = int(self.silence_duration * self.sample_rate / self.chunk_size)

        # Temp file management
        self.temp_dir = Path("/tmp/bluetooth_audio")
        self.temp_dir.mkdir(exist_ok=True)

        print("[BT Capturer] Initialized buffered capturer")

    async def start(self):
        """Start Bluetooth audio capture and processing."""
        print("[BT Capturer] Starting buffered capture")
        self.is_capturing = True

        # Detect and mute Bluetooth stream
        if not await self._detect_and_mute_stream():
            print("[BT Capturer ERROR] Could not detect/mute Bluetooth stream")
            return

        # Start capture process
        if not self._start_capture_process():
            print("[BT Capturer ERROR] Could not start parec capture")
            return

        # Start processing loop
        self.capture_task = asyncio.create_task(self._capture_loop())
        print("[BT Capturer] Capture loop started")

    async def stop(self):
        """Stop capture and cleanup."""
        print("[BT Capturer] Stopping")
        self.is_capturing = False

        # Cancel capture task
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass

        # Stop capture process
        if self.capture_process:
            self.capture_process.terminate()
            try:
                self.capture_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.capture_process.kill()
            self.capture_process = None

        # Unmute stream
        if self.bluetooth_stream_id:
            subprocess.run(["wpctl", "set-mute", str(self.bluetooth_stream_id), "0"])
            print(f"[BT Capturer] Unmuted stream {self.bluetooth_stream_id}")

        print("[BT Capturer] Stopped")

    async def _detect_and_mute_stream(self) -> bool:
        """Detect bluez_input stream and mute it.

        Returns:
            True if stream detected and muted, False otherwise
        """
        try:
            # Get wpctl status
            result = subprocess.run(
                ["wpctl", "status"],
                capture_output=True,
                text=True,
                check=True
            )

            # Find bluez_input stream ID
            for line in result.stdout.splitlines():
                if self.stream_name_pattern in line:
                    # Extract stream ID (format: "  82. bluez_input...")
                    parts = line.strip().split(".", 1)
                    if parts:
                        stream_id = parts[0].strip()
                        self.bluetooth_stream_id = int(stream_id)
                        break

            if not self.bluetooth_stream_id:
                print(f"[BT Capturer] No {self.stream_name_pattern} stream found")
                return False

            print(f"[BT Capturer] Found Bluetooth stream ID: {self.bluetooth_stream_id}")

            # Mute the stream
            subprocess.run(
                ["wpctl", "set-mute", str(self.bluetooth_stream_id), "1"],
                check=True
            )
            print(f"[BT Capturer] Muted stream {self.bluetooth_stream_id}")

            return True

        except Exception as e:
            print(f"[BT Capturer ERROR] Stream detection failed: {e}")
            return False

    def _start_capture_process(self) -> bool:
        """Start parec subprocess to capture from muted Bluetooth stream.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Capture from the specific Bluetooth stream monitor
            self.capture_process = subprocess.Popen(
                [
                    "parec",
                    f"--monitor-stream={self.bluetooth_stream_id}",
                    f"--format=s16le",
                    f"--rate={self.sample_rate}",
                    f"--channels={self.channels}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            print(f"[BT Capturer] Started parec capture from stream {self.bluetooth_stream_id}")
            return True

        except Exception as e:
            print(f"[BT Capturer ERROR] Failed to start parec: {e}")
            return False

    async def _capture_loop(self):
        """Main capture loop - buffer audio and detect sentence boundaries."""
        print("[BT Capturer] Capture loop running")

        try:
            while self.is_capturing:
                # Read audio chunk
                chunk_bytes = self.chunk_size * self.bytes_per_frame
                data = self.capture_process.stdout.read(chunk_bytes)

                if not data or len(data) < chunk_bytes:
                    await asyncio.sleep(0.01)
                    continue

                # Add to buffer
                self.audio_buffer.append(data)

                # Check if this chunk is silence
                if self._is_silence(data):
                    self.silence_frames += 1
                else:
                    self.silence_frames = 0

                # Sentence boundary detected?
                if self.silence_frames >= self.silence_frames_needed:
                    if len(self.audio_buffer) > 0:
                        await self._process_buffered_sentence()
                    self.audio_buffer = []
                    self.silence_frames = 0

                await asyncio.sleep(0.001)  # Prevent CPU spinning

        except asyncio.CancelledError:
            print("[BT Capturer] Capture loop cancelled")
        except Exception as e:
            print(f"[BT Capturer ERROR] Capture loop error: {e}")

    def _is_silence(self, data: bytes) -> bool:
        """Check if audio chunk is silence based on energy threshold.

        Args:
            data: Raw PCM audio data (s16le)

        Returns:
            True if chunk is silent, False otherwise
        """
        # Convert bytes to numpy array
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Calculate RMS energy
        energy = np.sqrt(np.mean(samples ** 2))

        return energy < self.silence_threshold

    async def _process_buffered_sentence(self):
        """Process completed sentence buffer - analyze and play with synchronized motion."""
        # Calculate sentence duration
        total_samples = len(self.audio_buffer) * self.chunk_size
        duration = total_samples / self.sample_rate

        # Skip very short sentences
        if duration < self.min_sentence_duration:
            print(f"[BT Capturer] Skipping short buffer ({duration:.2f}s)")
            return

        print(f"[BT Capturer] Processing sentence buffer ({duration:.2f}s)")

        # Save to temporary WAV file
        wav_path = self.temp_dir / f"bt_audio_{int(time.time() * 1000)}.wav"

        try:
            with wave.open(str(wav_path), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.bytes_per_sample)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_buffer))

            print(f"[BT Capturer] Saved WAV: {wav_path}")

            # Analyze audio with SpeechAnalyzer
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None,
                self.audio_coordinator.speech_analyzer.analyze,
                str(wav_path),
                ""  # No text available for Bluetooth audio
            )

            # Load motion timeline
            self.audio_coordinator.speech_offset_player.load_timeline(analysis)

            # Play audio + synchronized motion
            audio_start_time = time.time()

            # Start speech motion playback
            self.audio_coordinator.speech_offset_player.play(audio_start_time)

            # Play audio through speakers
            await self.audio_coordinator.audio_manager.play_audio(str(wav_path))

            # Stop speech motion
            self.audio_coordinator.speech_offset_player.stop()

            print(f"[BT Capturer] Playback complete")

            # Cleanup temp file
            wav_path.unlink()

        except Exception as e:
            print(f"[BT Capturer ERROR] Sentence processing failed: {e}")
            if wav_path.exists():
                wav_path.unlink()

    def get_state(self) -> dict:
        """Get current capturer state.

        Returns:
            State dictionary with capture status
        """
        return {
            "is_capturing": self.is_capturing,
            "stream_id": self.bluetooth_stream_id,
            "buffer_size": len(self.audio_buffer)
        }
