#!/usr/bin/env python3
"""
Silero VAD Wrapper - ML-based Voice Activity Detection

Uses Silero VAD model (ONNX) for robust speech detection without calibration.
Handles 512-sample chunks at 16kHz (32ms) and provides speech probability (0.0-1.0).
"""

import numpy as np
from typing import Optional
from pysilero_vad import SileroVoiceActivityDetector


class SileroVAD:
    """
    Wrapper for Silero VAD model with 512-sample chunk processing.

    Silero VAD is an ML-based voice activity detector that:
    - Works across 6000+ languages without tuning
    - Distinguishes speech from background noise at similar energy levels
    - Requires no calibration or environment-specific configuration
    - Outputs probability score (0.0-1.0) for each audio chunk

    Technical specs:
    - Sample rate: 16000 Hz
    - Chunk size: 512 samples (32ms)
    - Processing latency: ~1ms per chunk on Pi 500
    - Model size: ~10MB (downloaded on first use, then cached)
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        """
        Initialize Silero VAD model.

        Args:
            sample_rate: Audio sample rate (must be 16000 Hz)
            threshold: Speech probability threshold (0.0-1.0, default 0.5)
        """
        if sample_rate != 16000:
            raise ValueError(f"Silero VAD requires 16000 Hz sample rate, got {sample_rate}")

        self.sample_rate = sample_rate
        self.threshold = threshold

        print("[Silero VAD] Initializing model (may download ~10MB on first use)...")
        try:
            # Initialize ONNX-based Silero VAD
            self.model = SileroVoiceActivityDetector()
            print("[Silero VAD] Model loaded successfully")
        except Exception as e:
            print(f"[Silero VAD ERROR] Failed to load model: {e}")
            raise

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Process 512-sample int16 audio chunk and return speech probability.

        Args:
            audio_chunk: numpy array of int16 samples (512 samples for 16kHz)

        Returns:
            Speech probability (0.0-1.0)
            - 0.0 = definitely silence
            - 1.0 = definitely speech
            - Typical threshold: 0.5

        Raises:
            ValueError: If chunk is not 512 samples
        """
        if len(audio_chunk) != 512:
            raise ValueError(f"Silero VAD expects 512 samples, got {len(audio_chunk)}")

        try:
            # pysilero-vad expects raw PCM bytes (512 samples * 2 bytes = 1024 bytes)
            # Convert numpy int16 array to bytes
            audio_bytes = audio_chunk.tobytes()

            # Get probability from model
            speech_prob = self.model(audio_bytes)

            return float(speech_prob)

        except Exception as e:
            import traceback
            print(f"[Silero VAD ERROR] Error processing chunk: {e}")
            print(f"[Silero VAD ERROR] Traceback: {traceback.format_exc()}")
            print(f"[Silero VAD ERROR] Audio chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")
            # Return 0.0 (silence) on error to avoid false positives
            return 0.0

    def reset_states(self):
        """
        Reset model state between utterances.

        Silero VAD maintains internal LSTM state across chunks for context.
        Call this method when starting a new recording session to clear history.
        """
        try:
            self.model.reset_states()
        except AttributeError:
            # pysilero-vad may not expose reset_states in all versions
            # Model will reset naturally after processing new audio
            pass
