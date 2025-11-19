#!/usr/bin/env python3
"""Buffered Bluetooth audio capture using PipeWire/PulseAudio proxy sink.

Architecture:
1. Create a Virtual Null Sink (silent output)
2. Detect ONLY Bluetooth streams (ignore robot's own playback)
3. Route Bluetooth → Null Sink (silent, data preserved)
4. Capture from Null Sink Monitor (full audio data at 100% volume)
5. Buffer by sentence → Analyze → Play through Real Speaker with motion sync

This fixes the "recording silence" bug where muting the stream destroyed the audio data.
"""

import asyncio
import subprocess
import numpy as np
import wave
import time
import re
from pathlib import Path


class BluetoothBufferedCapturer:
    """Capture Bluetooth audio silently, buffer by sentence, then play with motion sync."""

    def __init__(self, audio_coordinator, movement_manager, state_tracker, timeout_callback):
        """Initialize buffered capturer.

        Args:
            audio_coordinator: AudioCoordinator instance for TTS playback and analysis
            movement_manager: MovementManager instance for motion control
            state_tracker: StateTracker instance for state management
            timeout_callback: Callback function to reset the master inactivity timeout
        """
        self.audio_coordinator = audio_coordinator
        self.movement_manager = movement_manager
        self.state_tracker = state_tracker
        self.timeout_callback = timeout_callback

        # Null sink configuration
        self.null_sink_name = "ReachyBluetoothProxy"
        self.null_sink_id = None
        self.monitor_source_name = None

        # Stream tracking
        self.bluetooth_stream_id = None
        self.own_playback_pids = set()  # Track our own playback processes

        # Audio capture
        self.capture_process = None
        self.capture_task = None
        self.is_capturing = False

        # Buffering parameters
        self.sample_rate = 16000
        self.channels = 2
        self.chunk_size = 2048
        self.bytes_per_sample = 2  # s16le
        self.bytes_per_frame = self.bytes_per_sample * self.channels

        # Sentence detection
        self.audio_buffer = []
        self.silence_threshold = 0.015  # Energy threshold for silence
        self.silence_duration = 1.2  # Seconds of silence = sentence boundary
        self.min_sentence_duration = 0.3  # Minimum sentence length (shorter for music phrases)
        self.silence_frames = 0
        self.silence_frames_needed = int(self.silence_duration * self.sample_rate / self.chunk_size)

        # Temp file management
        self.temp_dir = Path("/tmp/bluetooth_audio")
        self.temp_dir.mkdir(exist_ok=True)

        # Manager loop state
        self.is_active = False
        self._manager_task = None

        print("[BT Capturer] Initialized buffered capturer with proxy sink architecture")

    async def start(self):
        """Start the persistent Bluetooth manager loop."""
        if self._manager_task and not self._manager_task.done():
            print("[BT Capturer] Manager is already running.")
            return

        print("[BT Capturer] Starting manager loop.")
        self._setup_null_sink()
        self.is_active = True
        self._manager_task = asyncio.create_task(self._manager_loop())

    async def stop(self):
        """Stop the persistent Bluetooth manager loop and all subprocesses."""
        if not self._manager_task or self._manager_task.done():
            print("[BT Capturer] Manager is not running.")
            return

        print("[BT Capturer] Stopping manager loop.")
        self.is_active = False
        self._manager_task.cancel()
        try:
            await self._manager_task
        except asyncio.CancelledError:
            pass
        self._manager_task = None
        self._cleanup_null_sink()
        print("[BT Capturer] Manager stopped.")

    def _setup_null_sink(self):
        """Create a null sink using PulseAudio commands (PipeWire compatibility layer)."""
        try:
            # Unload any existing null sink first
            subprocess.run(
                ["pactl", "unload-module", "module-null-sink"],
                stderr=subprocess.DEVNULL,
                timeout=2
            )

            # Load null sink module
            result = subprocess.run(
                [
                    "pactl", "load-module", "module-null-sink",
                    f"sink_name={self.null_sink_name}",
                    f"sink_properties=device.description=Reachy_Bluetooth_Proxy"
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2
            )

            print(f"[BT Capturer] Created null sink: {self.null_sink_name}")

            # The monitor source name follows PulseAudio convention
            self.monitor_source_name = f"{self.null_sink_name}.monitor"

            # Verify the monitor source exists
            if self._verify_monitor_source():
                print(f"[BT Capturer] Verified monitor source: {self.monitor_source_name}")
            else:
                print(f"[BT Capturer WARNING] Monitor source not found, will retry during capture")

            # Find the sink ID for routing
            self._find_null_sink_id()

        except subprocess.TimeoutExpired:
            print("[BT Capturer ERROR] Null sink creation timed out")
        except subprocess.CalledProcessError as e:
            print(f"[BT Capturer ERROR] Failed to create null sink: {e}")
        except Exception as e:
            print(f"[BT Capturer ERROR] Null sink setup failed: {e}")

    def _verify_monitor_source(self) -> bool:
        """Verify that the monitor source exists."""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return self.monitor_source_name in result.stdout
        except:
            return False

    def _find_null_sink_id(self):
        """Find the sink ID of our null sink."""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True,
                text=True,
                timeout=2
            )
            for line in result.stdout.splitlines():
                if self.null_sink_name in line:
                    # Format: "ID    NAME    MODULE    SAMPLE_SPEC    STATE"
                    parts = line.split()
                    if parts:
                        self.null_sink_id = parts[0]
                        print(f"[BT Capturer] Found null sink ID: {self.null_sink_id}")
                        return
        except:
            pass

    def _cleanup_null_sink(self):
        """Remove the null sink."""
        try:
            subprocess.run(
                ["pactl", "unload-module", "module-null-sink"],
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            print("[BT Capturer] Cleaned up null sink")
        except:
            pass

    async def _manager_loop(self):
        """Main persistent loop that waits for, captures, and re-arms for Bluetooth streams."""
        consecutive_failures = 0
        MAX_FAILURES = 3

        try:
            while self.is_active:
                # PHASE 1: Wait for a Bluetooth stream to appear
                print("[BT Capturer] Waiting for a Bluetooth audio stream...")
                self.state_tracker.update_state("bluetooth_ready")
                self.bluetooth_stream_id = None

                while self.is_active and self.bluetooth_stream_id is None:
                    self._find_bluetooth_stream()
                    if self.bluetooth_stream_id is None:
                        await asyncio.sleep(1.0)  # Poll every second

                if not self.is_active:
                    break  # Exit if stop() was called

                # PHASE 2: Stream Found - Route to Null Sink and Start Capture
                print(f"[BT Capturer] Bluetooth stream {self.bluetooth_stream_id} detected.")

                if not self.null_sink_id:
                    self._find_null_sink_id()

                if not self.null_sink_id:
                    print("[BT Capturer ERROR] Null sink not found, cannot route stream.")
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_FAILURES:
                        print(f"[BT Capturer CRITICAL] Null sink not available after {MAX_FAILURES} attempts.")
                        break
                    await asyncio.sleep(2.0)
                    continue

                # Route Bluetooth stream to null sink
                if not self._route_stream_to_null_sink(self.bluetooth_stream_id):
                    print("[BT Capturer ERROR] Failed to route stream, returning to wait state.")
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_FAILURES:
                        print(f"[BT Capturer CRITICAL] Failed to route stream {MAX_FAILURES} times.")
                        break
                    await asyncio.sleep(2.0)
                    continue

                # Start capture process
                if not self._start_capture_process():
                    print("[BT Capturer ERROR] Failed to start capture, returning to wait state.")
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_FAILURES:
                        print(f"[BT Capturer CRITICAL] Failed to start capture {MAX_FAILURES} times.")
                        break
                    await asyncio.sleep(2.0)
                    continue

                # Reset failure counter on successful start
                consecutive_failures = 0

                # PHASE 3: Capture is running, start the processing loop
                self.is_capturing = True
                self.capture_task = asyncio.create_task(self._capture_loop())

                # PHASE 4: Monitor until the stream disappears
                while self.is_active and self._is_stream_active():
                    await asyncio.sleep(2.0)  # Check every 2 seconds

                print(f"[BT Capturer] Stream {self.bluetooth_stream_id} disappeared. Re-arming.")

                # PHASE 5: Cleanup and loop back to Phase 1
                self.is_capturing = False
                self._stop_capture_process()

                # Wait for capture task to finish processing any buffered sentence
                if self.capture_task and not self.capture_task.done():
                    await self.capture_task

        except asyncio.CancelledError:
            print("[BT Capturer] Manager loop cancelled.")
        finally:
            # Final cleanup
            self.is_capturing = False
            if self.capture_task and not self.capture_task.done():
                self.capture_task.cancel()
                try:
                    await self.capture_task
                except asyncio.CancelledError:
                    pass
            self._stop_capture_process()

    def _find_bluetooth_stream(self):
        """Find Bluetooth sink-input using pactl, IGNORING robot's own playback."""
        try:
            # List all sink-inputs (active streams)
            result = subprocess.run(
                ["pactl", "list", "sink-inputs"],
                capture_output=True,
                text=True,
                timeout=2
            )

            # Parse output to find Bluetooth streams
            # Format: Sink Input #123
            #         ...
            #         application.name = "bluez"
            #         ...

            current_id = None
            is_bluetooth = False

            for line in result.stdout.splitlines():
                # Look for "Sink Input #ID"
                match = re.match(r'Sink Input #(\d+)', line)
                if match:
                    # Save previous stream if it was Bluetooth
                    if current_id and is_bluetooth:
                        self.bluetooth_stream_id = int(current_id)
                        return

                    # Start tracking new stream
                    current_id = match.group(1)
                    is_bluetooth = False

                # Check if this stream is Bluetooth
                if current_id:
                    # Look for bluez in application name, process name, or client name
                    if 'application.name' in line or 'application.process.binary' in line:
                        if 'bluez' in line.lower():
                            is_bluetooth = True
                    # Also check media.name for Bluetooth indicator
                    if 'media.name' in line and 'bluetooth' in line.lower():
                        is_bluetooth = True

            # Check last stream
            if current_id and is_bluetooth:
                self.bluetooth_stream_id = int(current_id)

        except subprocess.TimeoutExpired:
            print("[BT Capturer ERROR] Finding Bluetooth stream timed out")
        except Exception as e:
            print(f"[BT Capturer ERROR] Failed to find Bluetooth stream: {e}")

    def _route_stream_to_null_sink(self, stream_id: int) -> bool:
        """Route a sink-input to the null sink using pactl."""
        try:
            subprocess.run(
                ["pactl", "move-sink-input", str(stream_id), self.null_sink_id],
                check=True,
                timeout=2
            )
            print(f"[BT Capturer] Routed stream {stream_id} to null sink {self.null_sink_id}")
            return True
        except subprocess.TimeoutExpired:
            print("[BT Capturer ERROR] Routing timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"[BT Capturer ERROR] Failed to route stream: {e}")
            return False

    def _is_stream_active(self) -> bool:
        """Check if the currently stored Bluetooth stream is still active."""
        if not self.bluetooth_stream_id:
            return False
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sink-inputs"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return f"{self.bluetooth_stream_id}\t" in result.stdout
        except:
            return False

    def _start_capture_process(self) -> bool:
        """Start parec subprocess to capture from null sink monitor.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if not self.monitor_source_name:
                print("[BT Capturer ERROR] Monitor source name not set")
                return False

            # Verify monitor source exists
            if not self._verify_monitor_source():
                print(f"[BT Capturer ERROR] Monitor source {self.monitor_source_name} not found")
                return False

            self.capture_process = subprocess.Popen(
                [
                    "parec",
                    f"--device={self.monitor_source_name}",
                    "--format=s16le",
                    f"--rate={self.sample_rate}",
                    f"--channels={self.channels}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            print(f"[BT Capturer] Started parec capture from {self.monitor_source_name}")
            return True

        except Exception as e:
            print(f"[BT Capturer ERROR] Failed to start parec: {e}")
            return False

    def _stop_capture_process(self):
        """Stop and clean up the parec subprocess."""
        if self.capture_process:
            self.capture_process.terminate()
            try:
                self.capture_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.capture_process.kill()
            self.capture_process = None

    async def _capture_loop(self):
        """Main capture loop - buffer audio and detect sentence boundaries."""
        print("[BT Capturer] Capture loop running")

        # Throttling for timeout resets
        last_timeout_reset = 0

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

                    # Reset timeout on active audio
                    current_time = time.time()
                    if current_time - last_timeout_reset > 5.0:
                        if self.timeout_callback:
                            self.timeout_callback()
                        last_timeout_reset = current_time

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
        # Reset the master inactivity timeout
        if self.timeout_callback:
            self.timeout_callback()

        # Calculate sentence duration
        total_samples = len(self.audio_buffer) * self.chunk_size
        duration = total_samples / self.sample_rate

        # Skip very short sentences
        if duration < self.min_sentence_duration:
            print(f"[BT Capturer] Skipping short buffer ({duration:.2f}s)")
            return

        print(f"[BT Capturer] Processing sentence buffer ({duration:.2f}s)")

        # Transition to bluetooth_playing state
        self.state_tracker.update_state("bluetooth_playing")

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

            # Start audio playback and get guaranteed start time
            # play_audio returns (start_time, completion_task) immediately
            audio_start_time, audio_playback_task = await self.audio_coordinator.audio_manager.play_audio(str(wav_path))

            # Start speech motion synchronized with audio start time
            self.audio_coordinator.speech_offset_player.play(audio_start_time)

            # Wait for audio playback to complete
            await audio_playback_task

            # Stop speech motion after audio completes
            self.audio_coordinator.speech_offset_player.stop()

            print(f"[BT Capturer] Playback complete")

            # Return to bluetooth_ready state after playback
            self.state_tracker.update_state("bluetooth_ready")

            # Cleanup temp file
            wav_path.unlink()

        except Exception as e:
            print(f"[BT Capturer ERROR] Sentence processing failed: {e}")
            import traceback
            traceback.print_exc()
            if wav_path.exists():
                wav_path.unlink()
            # Return to ready state even on error
            self.state_tracker.update_state("bluetooth_ready")

    def get_state(self) -> dict:
        """Get current capturer state.

        Returns:
            State dictionary with capture status
        """
        return {
            "is_capturing": self.is_capturing,
            "stream_id": self.bluetooth_stream_id,
            "buffer_size": len(self.audio_buffer),
            "null_sink_id": self.null_sink_id,
        }
