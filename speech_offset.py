#!/usr/bin/env python3
"""
Speech Offset Player for TTS Motion Synchronization

Generates and plays back motion timelines synchronized with TTS audio playback.
Motion offsets are layered on top of breathing as secondary offsets.

Based on audio analysis from speech_analyzer.py, this system maps:
- Pitch contour → Antenna z-offset (height)
- Energy peaks → Head bobs (pitch)
- Pauses → Reset to baseline
- Questions → Head tilt up
- Mood → Global amplitude scaling
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


# Removed mood scaling - movements use constant amplitude


class SpeechOffsetPlayer:
    """Plays back motion timelines synchronized with TTS audio."""

    def __init__(self, movement_manager: Any):
        """Initialize speech offset player.

        Args:
            movement_manager: MovementManager instance with set_speech_offsets method
        """
        self.movement_manager = movement_manager
        self._timeline: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._update_rate = 50  # Hz (20ms updates)

    def load_timeline(self, analysis: Dict[str, Any]) -> None:
        """Generate motion timeline from speech analysis.

        Args:
            analysis: Dict from speech_analyzer.analyze() with audio_features and text_features
        """
        logger.info("Generating motion timeline from speech analysis")
        self._timeline = self._generate_timeline(analysis)
        logger.info(f"Generated timeline with {len(self._timeline)} events over {analysis['duration']:.2f}s")

    def play(self, start_time: float) -> None:
        """Start playback thread synchronized to audio start time.

        Args:
            start_time: Timestamp from time.time() when audio playback begins
        """
        if self._thread and self._thread.is_alive():
            logger.warning("Playback already active, stopping previous playback")
            self.stop()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._playback_loop,
            args=(start_time,),
            daemon=True,
            name="SpeechOffsetPlayer"
        )
        self._thread.start()
        logger.info("Speech motion playback started")

    def stop(self) -> None:
        """Stop playback thread and reset offsets to zero."""
        if not self._thread or not self._thread.is_alive():
            # Reset offsets even if thread not running
            self.movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            return

        self._stop_event.set()
        self._thread.join(timeout=1.0)

        if self._thread.is_alive():
            logger.warning("Playback thread did not exit cleanly")

        # Reset offsets to zero
        self.movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        logger.info("Speech motion playback stopped, offsets reset")

    def _generate_timeline(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate motion timeline from analysis features.

        Args:
            analysis: Speech analysis dict

        Returns:
            List of timeline events sorted by time
        """
        events = []
        audio_feat = analysis.get('audio_features', {})
        text_feat = analysis.get('text_features', {})
        duration = analysis.get('duration', 0.0)

        baseline_pitch_rad = 0.0  # No baseline pitch offset during speech

        # 1. Generate oscillator events (antenna + head pitch, continuous throughout)
        oscillator_events = self._generate_pitch_events(audio_feat, baseline_pitch_rad)
        events.extend(oscillator_events)

        # 2. Peak-based nods disabled - using continuous pitch oscillator instead
        # events.extend(self._generate_peak_events(audio_feat, oscillator_events, baseline_pitch_rad))

        # 3. Generate pause → reset events (discrete)
        events.extend(self._generate_pause_events(audio_feat, baseline_pitch_rad))

        # 4. Generate question → head tilt event (end of speech)
        if text_feat.get('questions'):
            events.extend(self._generate_question_events(duration, baseline_pitch_rad))

        # Sort by time
        events.sort(key=lambda e: e['time'])

        # Ensure first event is at t=0 with zero offsets
        if not events or events[0]['time'] > 0.0:
            events.insert(0, {'time': 0.0, 'offsets': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'antennas': (0.0, 0.0)})

        return events

    def _generate_pitch_events(self, audio_feat: Dict, baseline_pitch_rad: float) -> List[Dict]:
        """Generate antenna rotation events from audio energy (loudness-driven oscillators).

        Args:
            audio_feat: Audio features dict

        Returns:
            List of timeline events with antenna rotations
        """
        events = []
        energy_envelope = audio_feat.get('energy_envelope', {})
        times = energy_envelope.get('times', [])
        values = energy_envelope.get('values', [])

        if not times or not values:
            logger.warning("No energy envelope data, skipping antenna motion")
            return events

        # Oscillator parameters
        antenna_freq_hz = 2.0  # Antenna roll oscillation frequency
        antenna_amplitude_rad = 1.5  # 1.5 radians (~86 degrees) amplitude

        pitch_freq_hz = 4.5  # Head pitch oscillation frequency (syllable rate)
        pitch_min_deg = -3.0  # Minimum pitch (slight tilt down)
        pitch_max_deg = 12.0  # Maximum pitch (tilt up)
        pitch_center_deg = (pitch_min_deg + pitch_max_deg) / 2.0  # Center at 5°
        pitch_amplitude_deg = (pitch_max_deg - pitch_min_deg) / 2.0  # ±10° range

        # Loudness normalization (RMS energy to gain)
        energy_min = float(np.percentile(values, 10))  # 10th percentile
        energy_max = float(np.percentile(values, 90))  # 90th percentile
        energy_range = energy_max - energy_min if energy_max > energy_min else 1.0

        logger.info(f"Energy range: {energy_min:.6f} - {energy_max:.6f}")

        # Random phase offsets for natural variation
        antenna_phase = np.random.random() * 2 * np.pi
        pitch_phase = np.random.random() * 2 * np.pi

        for t, energy in zip(times, values):
            # Normalize energy to [0, 1] loudness gain
            loudness = (energy - energy_min) / energy_range
            loudness = np.clip(loudness, 0.0, 1.0)

            # Antenna oscillator: Sine wave modulated by loudness
            antenna_angle = antenna_amplitude_rad * loudness * np.sin(2 * np.pi * antenna_freq_hz * t + antenna_phase)
            left_antenna = np.clip(antenna_angle, -3.1, 0.1)
            right_antenna = np.clip(-antenna_angle, -0.1, 3.1)

            # Head pitch oscillator: Sine wave modulated by loudness, asymmetric range
            pitch_oscillation = np.sin(2 * np.pi * pitch_freq_hz * t + pitch_phase)
            pitch_deg = pitch_center_deg + pitch_amplitude_deg * loudness * pitch_oscillation
            pitch_rad = np.deg2rad(pitch_deg)

            events.append({
                'time': float(t),
                'offsets': (0.0, 0.0, 0.0, 0.0, pitch_rad, 0.0),  # Pitch oscillation
                'antennas': (left_antenna, right_antenna)
            })

        logger.debug(f"Generated {len(events)} loudness-driven antenna oscillator events")
        return events

    def _generate_peak_events(self, audio_feat: Dict, pitch_events: List[Dict], baseline_pitch_rad: float) -> List[Dict]:
        """Generate head nod events from energy peaks, preserving Z and antenna values from pitch contour.

        Args:
            audio_feat: Audio features dict
            pitch_events: Voice pitch events to sample Z and antenna values from

        Returns:
            List of timeline events
        """
        events = []
        energy_stats = audio_feat.get('energy_stats', {})
        peak_times = energy_stats.get('peak_times', [])

        # Negative head pitch nod: -15 degrees (tilting head up toward user, subtle emphasis)
        nod_amplitude = -0.262  # -15° head pitch in radians (subtle emphasis)
        nod_duration = 0.4  # Hold nod for 400ms (snappier motion)
        nod_x_forward = 0.007  # 7mm forward movement during nod

        for peak_time in peak_times:
            # Sample Z and antenna values from pitch timeline at this moment
            z_value = 0.0
            antenna_values = (0.0, 0.0)

            # Find surrounding pitch events to interpolate Z and antennas
            for i in range(len(pitch_events) - 1):
                if pitch_events[i]['time'] <= peak_time <= pitch_events[i+1]['time']:
                    # Interpolate between these two events
                    t_range = pitch_events[i+1]['time'] - pitch_events[i]['time']
                    if t_range > 0:
                        t = (peak_time - pitch_events[i]['time']) / t_range
                        z_value = pitch_events[i]['offsets'][2] + t * (pitch_events[i+1]['offsets'][2] - pitch_events[i]['offsets'][2])
                        antenna_values = (
                            pitch_events[i]['antennas'][0] + t * (pitch_events[i+1]['antennas'][0] - pitch_events[i]['antennas'][0]),
                            pitch_events[i]['antennas'][1] + t * (pitch_events[i+1]['antennas'][1] - pitch_events[i]['antennas'][1])
                        )
                    else:
                        z_value = pitch_events[i]['offsets'][2]
                        antenna_values = pitch_events[i]['antennas']
                    break

            # Nod start (negative pitch = tilt head up, X forward = lean in, preserve Z and antennas)
            events.append({
                'time': float(peak_time),
                'offsets': (nod_x_forward, 0.0, z_value, 0.0, nod_amplitude, 0.0),
                'antennas': antenna_values
            })

            # Nod end (return to baseline pitch, preserve Z and antennas at end time)
            end_time = peak_time + nod_duration
            z_value_end = 0.0
            antenna_values_end = (0.0, 0.0)

            for i in range(len(pitch_events) - 1):
                if pitch_events[i]['time'] <= end_time <= pitch_events[i+1]['time']:
                    t_range = pitch_events[i+1]['time'] - pitch_events[i]['time']
                    if t_range > 0:
                        t = (end_time - pitch_events[i]['time']) / t_range
                        z_value_end = pitch_events[i]['offsets'][2] + t * (pitch_events[i+1]['offsets'][2] - pitch_events[i]['offsets'][2])
                        antenna_values_end = (
                            pitch_events[i]['antennas'][0] + t * (pitch_events[i+1]['antennas'][0] - pitch_events[i]['antennas'][0]),
                            pitch_events[i]['antennas'][1] + t * (pitch_events[i+1]['antennas'][1] - pitch_events[i]['antennas'][1])
                        )
                    else:
                        z_value_end = pitch_events[i]['offsets'][2]
                        antenna_values_end = pitch_events[i]['antennas']
                    break

            events.append({
                'time': float(end_time),
                'offsets': (0.0, 0.0, z_value_end, 0.0, 0.0, 0.0),
                'antennas': antenna_values_end
            })

        logger.debug(f"Generated {len(events)} head bob events from {len(peak_times)} peaks")
        return events

    def _generate_pause_events(self, audio_feat: Dict, baseline_pitch_rad: float) -> List[Dict]:
        """Generate reset events during pauses.

        Args:
            audio_feat: Audio features dict

        Returns:
            List of timeline events
        """
        events = []
        pauses = audio_feat.get('pauses', [])

        blend_duration = 0.2  # 200ms to blend to zero

        for pause in pauses:
            pause_start = pause['start']
            pause_end = pause['end']

            # Start blending to zero at pause start
            events.append({
                'time': float(pause_start),
                'offsets': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                'antennas': (0.0, 0.0)
            })

            # Hold zero during pause
            events.append({
                'time': float(pause_start + blend_duration),
                'offsets': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                'antennas': (0.0, 0.0)
            })

            # Resume motion after pause ends
            # (will be interpolated with surrounding pitch events)

        logger.debug(f"Generated {len(events)} pause reset events from {len(pauses)} pauses")
        return events

    def _generate_question_events(self, duration: float, baseline_pitch_rad: float) -> List[Dict]:
        """Generate head tilt event for questions.

        Args:
            duration: Total speech duration in seconds

        Returns:
            List of timeline events
        """
        # Tilt head up by 4° during last 0.5s of speech
        tilt_start = max(0.0, duration - 0.5)
        tilt_amplitude = -0.070  # -4° in radians (negative = look up)

        events = [
            {'time': tilt_start, 'offsets': (0.0, 0.0, 0.0, 0.0, tilt_amplitude, 0.0), 'antennas': (0.0, 0.0)},
            {'time': duration, 'offsets': (0.0, 0.0, 0.0, 0.0, tilt_amplitude, 0.0), 'antennas': (0.0, 0.0)},
        ]

        logger.debug(f"Generated question head tilt at {tilt_start:.2f}s")
        return events

    def _playback_loop(self, start_time: float) -> None:
        """Playback loop that interpolates and applies offsets.

        Args:
            start_time: Audio start timestamp from time.time()
        """
        logger.debug("Playback loop started")
        update_interval = 1.0 / self._update_rate

        while not self._stop_event.is_set():
            loop_start = time.time()

            # Calculate elapsed time since audio started
            elapsed = time.time() - start_time

            # Find surrounding timeline events for interpolation
            before = None
            after = None

            for event in self._timeline:
                if event['time'] <= elapsed:
                    before = event
                elif event['time'] > elapsed and after is None:
                    after = event
                    break

            # Interpolate offsets and antennas
            if before is None:
                # Before timeline starts - use zero
                offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                antennas = (0.0, 0.0)
            elif after is None:
                # After timeline ends - hold final values
                offsets = before['offsets']
                antennas = before.get('antennas', (0.0, 0.0))
            else:
                # Interpolate between before and after
                t_range = after['time'] - before['time']
                if t_range > 0:
                    t = (elapsed - before['time']) / t_range
                    offsets = tuple(
                        before['offsets'][i] + t * (after['offsets'][i] - before['offsets'][i])
                        for i in range(6)
                    )
                    # Interpolate antennas
                    before_ant = before.get('antennas', (0.0, 0.0))
                    after_ant = after.get('antennas', (0.0, 0.0))
                    antennas = (
                        before_ant[0] + t * (after_ant[0] - before_ant[0]),
                        before_ant[1] + t * (after_ant[1] - before_ant[1])
                    )
                else:
                    offsets = before['offsets']
                    antennas = before.get('antennas', (0.0, 0.0))

            # Apply offsets and antennas to movement manager
            try:
                # Debug: Print non-zero values
                if any(abs(v) > 0.001 for v in offsets) or any(abs(a) > 0.001 for a in antennas):
                    print(f"[Speech] t={elapsed:.2f}s pitch={np.rad2deg(offsets[4]):.1f}° z={offsets[2]*1000:.1f}mm ant=({np.rad2deg(antennas[0]):.1f}°, {np.rad2deg(antennas[1]):.1f}°)")
                self.movement_manager.set_speech_offsets(offsets, antennas)
            except Exception as e:
                logger.error(f"Failed to apply speech offsets: {e}")
                break

            # Sleep until next update
            sleep_time = update_interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

