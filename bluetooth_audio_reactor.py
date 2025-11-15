#!/usr/bin/env python3
"""
Bluetooth Audio Reactor for Real-Time Motion Synchronization

Captures Bluetooth audio stream from iPhone Claude app and generates
synchronized motion offsets based on real-time energy analysis.

Architecture:
- 300ms rolling buffer for audio capture and playback
- 50ms (20 Hz) energy analysis rate
- Motion synchronized with audio playback (both have 300ms delay)
- Energy-driven antenna oscillation (2 Hz) and head pitch (4.5 Hz)
"""

import time
import asyncio
import logging
import numpy as np
import pyaudio
from typing import Optional, Any
from collections import deque
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class BluetoothReactorState:
    """State tracking for Bluetooth audio reactor"""
    is_active: bool = False
    is_receiving_audio: bool = False
    bluetooth_device_name: Optional[str] = None
    current_intensity: float = 0.0
    energy_min: float = 0.001  # Running minimum for normalization
    energy_max: float = 0.1    # Running maximum for normalization


class BluetoothAudioReactor:
    """
    Real-time Bluetooth audio capture and motion synchronization.

    Monitors Bluetooth audio stream, analyzes energy in real-time,
    and applies synchronized motion offsets to movement manager.
    """

    def __init__(self, movement_manager: Any, audio_manager: Any):
        """
        Initialize Bluetooth audio reactor.

        Args:
            movement_manager: MovementManager instance with set_speech_offsets method
            audio_manager: AudioManager instance for playback coordination
        """
        self.movement_manager = movement_manager
        self.audio_manager = audio_manager

        # State tracking
        self.state = BluetoothReactorState()
        self._stop_event = asyncio.Event()
        self._reactor_task: Optional[asyncio.Task] = None

        # Audio configuration
        self.sample_rate = 22050  # Match SpeechAnalyzer sample rate
        self.buffer_duration_ms = 300  # 300ms rolling buffer
        self.analysis_interval_ms = 50  # 50ms analysis chunks (20 Hz)

        # Calculate buffer sizes
        self.buffer_size = int(self.sample_rate * self.buffer_duration_ms / 1000)
        self.chunk_size = int(self.sample_rate * self.analysis_interval_ms / 1000)

        # Rolling audio buffer (300ms)
        self.audio_buffer = deque(maxlen=self.buffer_size)

        # PyAudio instance
        self.pa = pyaudio.PyAudio()

        # Motion parameters (from SpeechOffsetPlayer)
        self.antenna_freq_hz = 2.0  # Antenna oscillation frequency
        self.antenna_amplitude_rad = 1.5  # Maximum antenna rotation (radians)

        self.pitch_freq_hz = 4.5  # Head pitch oscillation frequency
        self.pitch_min_deg = -3.0  # Minimum pitch (slight down)
        self.pitch_max_deg = 12.0  # Maximum pitch (up)
        self.pitch_center_deg = (self.pitch_min_deg + self.pitch_max_deg) / 2.0
        self.pitch_amplitude_deg = (self.pitch_max_deg - self.pitch_min_deg) / 2.0

        # Silence detection
        self.silence_threshold = 0.01  # RMS threshold for silence
        self.silence_duration_for_reset = 0.5  # 500ms of silence → reset to neutral
        self.last_sound_time = time.time()

        # Energy normalization (running percentiles)
        self.energy_window_size = 100  # Keep last 100 energy measurements
        self.energy_history = deque(maxlen=self.energy_window_size)

        # Phase offsets for natural variation
        self.antenna_phase = np.random.random() * 2 * np.pi
        self.pitch_phase = np.random.random() * 2 * np.pi

        logger.info(f"BluetoothAudioReactor initialized:")
        logger.info(f"  Sample rate: {self.sample_rate} Hz")
        logger.info(f"  Buffer size: {self.buffer_duration_ms}ms ({self.buffer_size} samples)")
        logger.info(f"  Analysis interval: {self.analysis_interval_ms}ms ({self.chunk_size} samples)")
        logger.info(f"  Antenna: {self.antenna_freq_hz} Hz, {self.antenna_amplitude_rad} rad max")
        logger.info(f"  Head pitch: {self.pitch_freq_hz} Hz, {self.pitch_min_deg}° to {self.pitch_max_deg}°")

    def _calculate_rms_energy(self, audio_chunk: np.ndarray) -> float:
        """
        Calculate Root Mean Square (RMS) energy of audio chunk.

        Args:
            audio_chunk: Numpy array of audio samples

        Returns:
            RMS energy value
        """
        return np.sqrt(np.mean(audio_chunk ** 2))

    def _normalize_intensity(self, energy: float) -> float:
        """
        Normalize energy to 0.0-1.0 intensity using running percentiles.

        Args:
            energy: Raw RMS energy value

        Returns:
            Normalized intensity (0.0 = silence, 1.0 = peak loudness)
        """
        # Add to history
        self.energy_history.append(energy)

        # Update running percentiles (10th and 90th)
        if len(self.energy_history) >= 10:
            self.state.energy_min = float(np.percentile(list(self.energy_history), 10))
            self.state.energy_max = float(np.percentile(list(self.energy_history), 90))

        # Normalize
        energy_range = self.state.energy_max - self.state.energy_min
        if energy_range > 0:
            intensity = (energy - self.state.energy_min) / energy_range
        else:
            intensity = 0.0

        return np.clip(intensity, 0.0, 1.0)

    def _calculate_motion_offsets(self, intensity: float, current_time: float) -> tuple:
        """
        Calculate motion offsets from intensity value.

        Args:
            intensity: Normalized intensity (0.0-1.0)
            current_time: Current timestamp for oscillator phase

        Returns:
            Tuple of (offsets, antennas) for movement_manager.set_speech_offsets()
        """
        # Check for silence
        if intensity < self.silence_threshold:
            if (current_time - self.last_sound_time) > self.silence_duration_for_reset:
                # Reset to neutral
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0)
        else:
            self.last_sound_time = current_time

        # Apply non-linear easing curve for punchier motion
        eased_intensity = intensity ** 2.0

        # Antenna oscillator: Sine wave modulated by loudness
        antenna_angle = self.antenna_amplitude_rad * eased_intensity * np.sin(
            2 * np.pi * self.antenna_freq_hz * current_time + self.antenna_phase
        )
        left_antenna = np.clip(antenna_angle, -3.1, 0.1)
        right_antenna = np.clip(-antenna_angle, -0.1, 3.1)

        # Head pitch oscillator: Sine wave modulated by loudness, asymmetric range
        pitch_oscillation = np.sin(2 * np.pi * self.pitch_freq_hz * current_time + self.pitch_phase)
        pitch_deg = self.pitch_center_deg + self.pitch_amplitude_deg * eased_intensity * pitch_oscillation
        pitch_rad = np.deg2rad(pitch_deg)

        # Return offsets (x, y, z, roll, pitch, yaw) and antennas (left, right)
        offsets = (0.0, 0.0, 0.0, 0.0, pitch_rad, 0.0)
        antennas = (left_antenna, right_antenna)

        return offsets, antennas


    async def _reactor_loop(self):
        """
        Main reactor loop: capture audio → analyze energy → apply motion.
        Uses 'parec' subprocess for robust PipeWire monitor capture.

        Runs at 20 Hz (50ms intervals) for motion updates.
        """
        logger.info("Starting Bluetooth audio reactor loop with parec subprocess")

        # The exact monitor source name proven to work
        monitor_source = "alsa_output.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_202000386253800122-00.analog-stereo.monitor"
        self.state.bluetooth_device_name = monitor_source

        # Command to capture raw 16-bit stereo audio at our desired sample rate
        parec_command = [
            "parec",
            f"--device={monitor_source}",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            "--channels=2",
        ]

        process = None
        self.state.is_active = True
        update_interval = self.analysis_interval_ms / 1000.0  # 50ms = 0.05s

        try:
            # Launch parec as a background process, piping its stdout to us
            process = await asyncio.create_subprocess_exec(
                *parec_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info(f"parec subprocess started, capturing from {monitor_source}")

            # Calculate how many bytes to read per chunk for 16-bit stereo
            bytes_per_sample = 2 * 2  # 2 bytes/sample, 2 channels
            chunk_size_bytes = self.chunk_size * bytes_per_sample

            while not self._stop_event.is_set():
                loop_start = time.time()

                # Read raw audio bytes directly from the parec subprocess
                try:
                    # Use readexactly to ensure we get full chunk
                    audio_bytes = await process.stdout.readexactly(chunk_size_bytes)
                except asyncio.IncompleteReadError as e:
                    # Parec stream ended or paused - if we got some data, use it
                    if len(e.partial) > 0:
                        logger.debug(f"Got partial chunk: {len(e.partial)} of {chunk_size_bytes} bytes")
                        audio_bytes = e.partial
                    else:
                        logger.warning(f"parec stream ended")
                        # Try to read stderr to see what went wrong
                        try:
                            stderr_bytes = await asyncio.wait_for(process.stderr.read(), timeout=0.5)
                            if stderr_bytes:
                                logger.error(f"parec stderr: {stderr_bytes.decode()}")
                        except:
                            pass
                        break
                except Exception as e:
                    logger.error(f"Error reading from parec: {e}")
                    break

                try:

                    # Convert raw 16-bit integer bytes into a float32 numpy array
                    audio_s16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float32 = audio_s16.astype(np.float32) / 32768.0

                    # Reshape to stereo and mix down to mono
                    audio_data_stereo = audio_float32.reshape(-1, 2)
                    audio_chunk = np.mean(audio_data_stereo, axis=1)

                    # Add to rolling buffer
                    self.audio_buffer.extend(audio_chunk)

                    # Calculate energy
                    energy = self._calculate_rms_energy(audio_chunk)

                    # Normalize to intensity
                    intensity = self._normalize_intensity(energy)
                    self.state.current_intensity = intensity

                    # Calculate motion offsets
                    current_time = time.time()
                    offsets, antennas = self._calculate_motion_offsets(intensity, current_time)

                    # Apply to movement manager
                    self.movement_manager.set_speech_offsets(offsets, antennas)

                    # Debug output (only for non-zero values)
                    if intensity > 0.01:
                        print(f"[BT Audio] Intensity: {intensity:.2f} | "
                              f"Pitch: {np.rad2deg(offsets[4]):.1f}° | "
                              f"Ant: ({np.rad2deg(antennas[0]):.1f}°, {np.rad2deg(antennas[1]):.1f}°)")
                        logger.debug(
                            f"Energy: {energy:.4f} → Intensity: {intensity:.2f} | "
                            f"Pitch: {np.rad2deg(offsets[4]):.1f}° | "
                            f"Ant: ({np.rad2deg(antennas[0]):.1f}°, {np.rad2deg(antennas[1]):.1f}°)"
                        )

                    # Mark receiving audio if intensity above threshold
                    self.state.is_receiving_audio = intensity > self.silence_threshold

                except Exception as e:
                    logger.error(f"Error in reactor loop iteration: {e}")

                # Sleep until next update (maintain 20 Hz rate)
                elapsed = time.time() - loop_start
                sleep_time = max(0, update_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except FileNotFoundError:
            logger.error("`parec` command not found. Please ensure it is installed.")
        except Exception as e:
            logger.error(f"Error in parec reactor loop: {e}")
            # Log stderr from the process if it exists
            if process and process.stderr:
                try:
                    stderr = await process.stderr.read()
                    if stderr:
                        logger.error(f"parec stderr: {stderr.decode()}")
                except:
                    pass
        finally:
            # Cleanup parec subprocess
            if process:
                logger.info("Terminating parec subprocess")
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("parec did not terminate, killing")
                    process.kill()
                    await process.wait()
                except:
                    pass

            self.state.is_active = False
            self.state.is_receiving_audio = False

            # Reset motion offsets to zero
            self.movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0))
            logger.info("Bluetooth audio reactor loop stopped, offsets reset")

    async def start(self):
        """Start Bluetooth audio reactor."""
        if self._reactor_task and not self._reactor_task.done():
            logger.warning("Reactor already running")
            return

        self._stop_event.clear()
        self._reactor_task = asyncio.create_task(self._reactor_loop())
        logger.info("Bluetooth audio reactor started")

    async def stop(self):
        """Stop Bluetooth audio reactor."""
        if not self._reactor_task or self._reactor_task.done():
            logger.warning("Reactor not running")
            return

        self._stop_event.set()
        await self._reactor_task
        self._reactor_task = None
        logger.info("Bluetooth audio reactor stopped")

    def get_state(self) -> dict:
        """Get current reactor state."""
        return {
            'is_active': self.state.is_active,
            'is_receiving_audio': self.state.is_receiving_audio,
            'bluetooth_device_name': self.state.bluetooth_device_name,
            'current_intensity': self.state.current_intensity,
            'energy_range': (self.state.energy_min, self.state.energy_max)
        }
