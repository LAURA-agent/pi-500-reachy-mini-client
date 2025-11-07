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


# Mood scaling factors
MOOD_SCALES = {
    None: 1.0,
    "calm": 0.7,
    "thoughtful": 0.8,
    "welcoming": 1.0,
    "celebratory": 1.5,
    "confused": 1.2,
    "frustrated": 1.5,
    "surprised": 1.8,
    "energetic": 1.8,
    "playful": 1.6,
}


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

        # Extract mood scaling factor
        mood = text_feat.get('mood')
        mood_scale = MOOD_SCALES.get(mood, 1.0)
        logger.debug(f"Mood: {mood}, scale: {mood_scale}")

        # 1. Generate pitch → antenna events (continuous throughout)
        events.extend(self._generate_pitch_events(audio_feat, mood_scale))

        # 2. Generate energy peak → head bob events (discrete)
        events.extend(self._generate_peak_events(audio_feat, mood_scale))

        # 3. Generate pause → reset events (discrete)
        events.extend(self._generate_pause_events(audio_feat))

        # 4. Generate question → head tilt event (end of speech)
        if text_feat.get('questions'):
            events.extend(self._generate_question_events(duration))

        # Sort by time
        events.sort(key=lambda e: e['time'])

        # Ensure first event is at t=0 with zero offsets
        if not events or events[0]['time'] > 0.0:
            events.insert(0, {'time': 0.0, 'offsets': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)})

        return events

    def _generate_pitch_events(self, audio_feat: Dict, mood_scale: float) -> List[Dict]:
        """Generate antenna z-offset events from pitch contour.

        Args:
            audio_feat: Audio features dict
            mood_scale: Mood amplitude scaling factor

        Returns:
            List of timeline events
        """
        events = []
        pitch_contour = audio_feat.get('pitch_contour', {})
        times = pitch_contour.get('times', [])
        values = pitch_contour.get('values', [])
        confidence = pitch_contour.get('confidence', [])

        if not times or not values or not confidence:
            logger.warning("No pitch contour data, skipping antenna motion")
            return events

        # Pitch range normalization: 160-350 Hz → 0.0-0.003 m (0-3mm)
        pitch_min = 160.0
        pitch_max = 350.0
        antenna_min = 0.0
        antenna_max = 0.003 * mood_scale  # Scale by mood

        for t, pitch, conf in zip(times, values, confidence):
            # Only use high-confidence voiced regions
            if conf < 0.5 or pitch < pitch_min:
                antenna_z = 0.0
            else:
                # Normalize and clamp
                normalized = (pitch - pitch_min) / (pitch_max - pitch_min)
                normalized = np.clip(normalized, 0.0, 1.0)
                antenna_z = antenna_min + normalized * (antenna_max - antenna_min)

            events.append({
                'time': float(t),
                'offsets': (0.0, 0.0, antenna_z, 0.0, 0.0, 0.0)
            })

        logger.debug(f"Generated {len(events)} pitch-based antenna events")
        return events

    def _generate_peak_events(self, audio_feat: Dict, mood_scale: float) -> List[Dict]:
        """Generate head bob events from energy peaks.

        Args:
            audio_feat: Audio features dict
            mood_scale: Mood amplitude scaling factor

        Returns:
            List of timeline events
        """
        events = []
        energy_stats = audio_feat.get('energy_stats', {})
        peak_times = energy_stats.get('peak_times', [])

        bob_amplitude = 0.02 * mood_scale  # 0.02 rad (~1.1°) pitch offset
        bob_duration = 0.1  # Hold bob for 100ms

        for peak_time in peak_times:
            # Bob start (pitch offset applied)
            events.append({
                'time': float(peak_time),
                'offsets': (0.0, 0.0, 0.0, 0.0, bob_amplitude, 0.0)
            })

            # Bob end (return to baseline)
            events.append({
                'time': float(peak_time + bob_duration),
                'offsets': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            })

        logger.debug(f"Generated {len(events)} head bob events from {len(peak_times)} peaks")
        return events

    def _generate_pause_events(self, audio_feat: Dict) -> List[Dict]:
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
                'offsets': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            })

            # Hold zero during pause
            events.append({
                'time': float(pause_start + blend_duration),
                'offsets': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            })

            # Resume motion after pause ends
            # (will be interpolated with surrounding pitch events)

        logger.debug(f"Generated {len(events)} pause reset events from {len(pauses)} pauses")
        return events

    def _generate_question_events(self, duration: float) -> List[Dict]:
        """Generate head tilt event for questions.

        Args:
            duration: Total speech duration in seconds

        Returns:
            List of timeline events
        """
        # Tilt head up by 5° during last 0.5s of speech
        tilt_start = max(0.0, duration - 0.5)
        tilt_amplitude = 0.087  # 5° in radians

        events = [
            {'time': tilt_start, 'offsets': (0.0, 0.0, 0.0, 0.0, tilt_amplitude, 0.0)},
            {'time': duration, 'offsets': (0.0, 0.0, 0.0, 0.0, tilt_amplitude, 0.0)},
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

            # Interpolate offsets
            if before is None:
                # Before timeline starts - use zero
                offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif after is None:
                # After timeline ends - hold final offsets
                offsets = before['offsets']
            else:
                # Interpolate between before and after
                t_range = after['time'] - before['time']
                if t_range > 0:
                    t = (elapsed - before['time']) / t_range
                    offsets = tuple(
                        before['offsets'][i] + t * (after['offsets'][i] - before['offsets'][i])
                        for i in range(6)
                    )
                else:
                    offsets = before['offsets']

            # Apply offsets to movement manager
            try:
                self.movement_manager.set_speech_offsets(offsets)
            except Exception as e:
                logger.error(f"Failed to apply speech offsets: {e}")
                break

            # Sleep until next update
            sleep_time = update_interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Playback loop exited")
