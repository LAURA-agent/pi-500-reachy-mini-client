#!/usr/bin/env python3
"""
Speech Processor - Manages VAD-based and push-to-talk speech capture

CRITICAL TIMING COORDINATION:
This module resolves conflicts between Silero VAD and VOSK's internal silence
detection. The key principle is:

    VAD CONTROLS TIMING → VOSK PROVIDES TRANSCRIPTION

VOSK TIMING ISSUES:
- VOSK has hardcoded internal timeouts (0.5s, 1.0s, 2.0s, 5.0s)
- These timeouts conflict with natural speech patterns
- VOSK sends premature "finals" that interrupt continuous speech
- Our VAD uses Silero ML model + frame counting for better accuracy

SILERO VAD:
- ML-based speech detection (no calibration needed)
- Processes 512-sample chunks (32ms) at 16kHz
- Outputs probability (0.0-1.0), threshold at 0.5
- Robust to background noise and environment changes

SOLUTION:
- VAD mode: Ignore VOSK finals, accumulate partials, use server's complete result
- Push-to-talk: User controls timing, same transcription strategy
- Both modes prioritize server's final_result over client fragments

TRANSCRIPT PRIORITY ORDER:
1. Server's final_result (VOSK's FinalResult() - most complete)
2. Accumulated partials (real-time transcription during capture)
3. Client's fragmented finals (last resort)
"""

import asyncio
import time
import traceback
import select
import platform
import numpy as np
from typing import Optional

# Optional evdev import (Linux-only)
try:
    from evdev import ecodes
except ImportError:
    # Mock ecodes for non-Linux platforms
    class ecodes:
        KEY_LEFTMETA = 125

from speech_capture.vosk_readiness_checker import vosk_readiness
from speech_capture.silero_vad_wrapper import SileroVAD


class SpeechProcessor:
    """
    Handles all speech capture and processing operations.
    
    Manages both VAD-based capture and push-to-talk modes with proper
    transcription coordination and manual stop handling.
    """
    
    def __init__(self, audio_manager, transcriber, keyboard_device=None):
        self.audio_manager = audio_manager
        self.transcriber = transcriber
        self.keyboard_device = keyboard_device

        # Initialize Silero VAD for ML-based speech detection
        print("[Speech Processor] Initializing Silero VAD...")
        self.silero_vad = SileroVAD(sample_rate=16000, threshold=0.5)
        print("[Speech Processor] Silero VAD ready")
        
    async def _check_manual_vad_stop(self):
        """Check for manual VAD stop via keyboard"""
        if not ecodes: 
            return False
            
        # Handle channel switching - refresh keyboard device if needed
        if not self.keyboard_device or not self._is_device_valid():
            self._refresh_keyboard_device()
            
        if not self.keyboard_device:
            return False
            
        try:
            if select.select([self.keyboard_device.fd], [], [], 0)[0]:
                for event in self.keyboard_device.read():
                    if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_LEFTMETA and event.value == 1:
                        print("[VAD] Manual stop via keyboard.")
                        return True
        except (BlockingIOError, OSError) as e:
            # Handle channel switching or device disconnection
            if "No such device" in str(e) or e.errno == 19:
                print(f"[VAD] Keyboard device changed (channel switch?), refreshing...")
                self._refresh_keyboard_device()
            else:
                print(f"[VAD] Keyboard read error: {e}")
        return False
        
    def _is_device_valid(self):
        """Check if keyboard device is still valid"""
        if not self.keyboard_device:
            return False
        try:
            # Quick test read with no blocking
            select.select([self.keyboard_device.fd], [], [], 0)
            return True
        except (OSError, ValueError):
            return False
            
    def _refresh_keyboard_device(self):
        """Refresh keyboard device connection for channel switching"""
        try:
            import evdev
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            
            # Look for K780 or any keyboard device
            for device in devices:
                if ('k780' in device.name.lower() or 
                    ('keyboard' in device.name.lower() and 'mouse' not in device.name.lower())):
                    # Close old device if exists
                    if self.keyboard_device:
                        try:
                            self.keyboard_device.close()
                        except:
                            pass
                    
                    self.keyboard_device = device
                    print(f"[VAD] Refreshed keyboard device: {device.name}")
                    return
                    
            # Keyboard not found - continue without logging to reduce noise
            self.keyboard_device = None
            
        except Exception as e:
            print(f"[VAD] Error refreshing keyboard device: {e}")
            self.keyboard_device = None

    async def capture_speech_with_unified_vad(self, display_manager, is_follow_up=False, claude_code_mode=False, immediate_feedback_callback=None) -> str | None:
        """Unified VAD function for speech capture"""
        # Check VOSK readiness before starting speech capture
        if not vosk_readiness.is_speech_enabled():
            print("[WARNING] VOSK server not ready - speech capture disabled")
            return None
            
        # start_listening() already waits for audio playback to complete
        audio_stream = await self.audio_manager.start_listening()
        
        if not audio_stream:
            print("[ERROR] Failed to start audio stream for VAD.")
            return None

        # Load VAD settings
        try:
            from tools.vad_settings import load_vad_settings
            vad_config = load_vad_settings()
            silero_threshold = vad_config.get('silero_threshold', 0.5)
            print(f"[VAD] Using Silero settings: threshold={silero_threshold:.2f}")
        except Exception as e:
            print(f"[VAD] Error loading settings, using defaults: {e}")
            from config.client_config import VAD_SETTINGS
            vad_config = VAD_SETTINGS
            silero_threshold = vad_config.get('silero_threshold', 0.5)

        # Extract parameters (Silero-specific)
        # Use 3.5 second silence timeout for all modes
        silence_duration = 3.5  # Optimal timeout for natural speech patterns
        min_speech_duration = vad_config.get('min_speech_duration', 0.4)
        speech_buffer_time = vad_config.get('speech_buffer_time', 1.0)
        max_recording_time = vad_config.get('max_recording_time', 45.0)
        print(f"[VAD DEBUG] max_recording_time from config: {max_recording_time}, full config keys: {list(vad_config.keys())}")

        # Timeout settings based on context - time to START speaking
        # Longer timeout for Claude Code mode to allow time to start speaking
        if claude_code_mode:
            initial_timeout = 7.0 if is_follow_up else 5.0
        else:
            initial_timeout = 5.0 if is_follow_up else 3.0

        print(f"[VAD] Starting {'follow-up' if is_follow_up else 'initial'} listening with {initial_timeout:.1f}s timeout")

        # Initialize transcriber connection at start of capture
        try:
            self.transcriber.reset()
        except Exception as e:
            print(f"[VAD] Transcriber reset failed: {e}")
            return None

        # Initialize state variables
        overall_start_time = time.monotonic()
        speech_start_time = 0
        voice_started = False
        silence_frames_count = 0
        last_partial_text = ""  # Track accumulated partials for fallback

        # Calculate frame timing (2048 samples at 16kHz = 128ms per frame)
        frames_per_second = self.audio_manager.sample_rate / self.audio_manager.frame_length
        silence_frames_needed = int(silence_duration * frames_per_second)

        # Reset Silero VAD state for new capture session
        self.silero_vad.reset_states()
        
        try:
            while True:
                current_time = time.monotonic()
                
                # Check for manual stop via keyboard
                if await self._check_manual_vad_stop():
                    print(f"[VAD] Manual stop triggered. Voice started: {voice_started}, Duration: {(current_time - speech_start_time) if voice_started else 0:.2f}s")
                    # Always process what we have, don't reset
                    if voice_started:
                        # Give a small buffer time to process the last audio
                        await asyncio.sleep(speech_buffer_time)
                    # Break without resetting - we want to keep any partial transcript
                    break

                # Timeout checks
                if not voice_started and (current_time - overall_start_time > initial_timeout):
                    elapsed = current_time - overall_start_time
                    print(f"[VAD] {'Follow-up' if is_follow_up else 'Initial'} timeout ({initial_timeout:.1f}s) reached at {elapsed:.2f}s. No voice detected.")
                    self.transcriber.reset()  # Clean up before returning
                    return None
                    
                if voice_started and (current_time - speech_start_time > max_recording_time):
                    print(f"[VAD] Max recording time ({max_recording_time:.1f}s) reached.")
                    break

                # Read audio frame
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.005)
                    continue

                # Validate frame size
                expected_frame_size = self.audio_manager.frame_length * 2
                if len(pcm_bytes) != expected_frame_size:
                    continue

                # Process with Silero VAD (subdivide 2048-sample frame into 4x 512-sample chunks)
                frame_data_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

                # Split 2048-sample frame into 4 chunks of 512 samples each for Silero
                chunks = [frame_data_int16[i:i+512] for i in range(0, 2048, 512)]
                speech_probabilities = [self.silero_vad.process_chunk(chunk) for chunk in chunks]

                # Use maximum probability across all chunks in this frame
                max_speech_probability = max(speech_probabilities)

                # Silero VAD State Machine - ML-based speech detection
                if not voice_started:

                    # Check if Silero probability is above threshold to start voice detection
                    if max_speech_probability > silero_threshold:
                        voice_started = True
                        speech_start_time = current_time
                        silence_frames_count = 0
                        elapsed = current_time - overall_start_time
                        print(f"[VAD] Voice started at {elapsed:.2f}s with Silero probability {max_speech_probability:.3f}")
                else:
                    # Voice has started - now process with VOSK for transcription
                    # ==================== VOSK TRANSCRIPTION PROCESSING ====================
                    # CRITICAL: Only send audio to VOSK after speech detection
                    # This prevents VOSK from hallucinating on pre-speech silence
                    try:
                        is_final_vosk, is_speech, partial_text_vosk = self.transcriber.process_frame(pcm_bytes)

                        # IGNORE VOSK's internal final decisions - they're premature
                        # (no print needed - this happens frequently and is expected)

                        # ACCUMULATE partial transcriptions during speech capture
                        # Keep the LONGEST partial seen - VOSK sometimes returns shorter versions
                        if is_speech and partial_text_vosk:
                            current_partial = partial_text_vosk.strip()
                            if len(current_partial) > len(last_partial_text):
                                last_partial_text = current_partial

                                # Display partial results (throttled to every 3 seconds)
                                if not hasattr(self, "last_partial_print_time") or (current_time - getattr(self, "last_partial_print_time", 0) > 3.0):
                                    print(f"[VOSK] Partial received: {current_partial}")
                                    self.last_partial_print_time = current_time

                    except Exception as vosk_error:
                        print(f"[VAD] Vosk processing error: {vosk_error}")

                    # Voice has started - check for continuation using same threshold
                    if max_speech_probability > silero_threshold:
                        silence_frames_count = 0
                    else:
                        silence_frames_count += 1

                    speech_duration = current_time - speech_start_time

                    # End conditions - ONLY use 3.5-second silence detection
                    if silence_frames_count >= silence_frames_needed and speech_duration >= min_speech_duration:
                        print(f"[VAD] End of speech by 3.5s silence. Duration: {speech_duration:.2f}s")

                        # Play immediate feedback if provided (for Claude Code)
                        if immediate_feedback_callback:
                            print("[VAD] Triggering immediate feedback...")
                            await immediate_feedback_callback()

                        # Continue processing for buffer_time to catch final VOSK partials
                        buffer_start = time.monotonic()
                        while time.monotonic() - buffer_start < speech_buffer_time:
                            # Read any remaining frames
                            pcm_bytes = self.audio_manager.read_audio_frame()
                            if pcm_bytes and len(pcm_bytes) == expected_frame_size:
                                try:
                                    is_final_vosk, is_speech, partial_text_vosk = self.transcriber.process_frame(pcm_bytes)

                                    # Continue accumulating partials during buffer period
                                    if is_speech and partial_text_vosk:
                                        current_partial = partial_text_vosk.strip()
                                        if len(current_partial) > len(last_partial_text):
                                            last_partial_text = current_partial
                                except Exception as e:
                                    print(f"[VAD] Buffer period error: {e}")

                            await asyncio.sleep(0.01)  # Small delay between reads
                        break

            # ==================== FINAL TRANSCRIPT EXTRACTION ====================
            # When VAD determines speech has ended, get the most complete transcription
            # 
            # TRANSCRIPT PRIORITY ORDER:
            # 1. Server's final_result (most complete - includes all processed audio)
            # 2. Accumulated partials (real-time transcription, often more accurate)
            # 3. Client's incomplete finals (last resort)
            #
            # The server's final_result is generated by calling VOSK's FinalResult() 
            # which consolidates everything processed during the session
            final_transcript = self.transcriber.get_final_text()
            print(f"[VAD] Raw final transcript from VOSK: '{final_transcript}'")
            
            # FALLBACK LOGIC: Use accumulated partials if VOSK final is incomplete
            # This happens when VOSK's internal state is fragmented due to timing conflicts
            if (not final_transcript or not final_transcript.strip()) and last_partial_text:
                print(f"[VAD] VOSK final empty/incomplete, using accumulated partials: '{last_partial_text}'")
                final_transcript = last_partial_text
            elif last_partial_text and len(last_partial_text) > len(final_transcript):
                print(f"[VAD] Accumulated partials longer than VOSK final - using partials: '{last_partial_text}'")
                final_transcript = last_partial_text

            if final_transcript:
                final_transcript = final_transcript.strip()
                
                # Apply filtering logic
                if not final_transcript:
                    print("[VAD] Transcript empty after stripping.")
                    return None

                # Background noise filter - reject if transcript is just "the" (common noise artifact)
                if final_transcript.lower().strip() == "the":
                    print(f"[VAD] Rejecting background noise artifact: '{final_transcript}'")
                    return None

                num_words = len(final_transcript.split())
                min_chars_single = 2
                min_words_overall = 1
                min_transcript_length = 2

                # Validation checks
                if num_words == 0 or len(final_transcript) < min_transcript_length:
                    print(f"[VAD] Rejecting (too short: {len(final_transcript)} chars): '{final_transcript}'")
                    return None
                    
                if num_words < min_words_overall:
                    print(f"[VAD] Rejecting (too few words: {num_words}): '{final_transcript}'")
                    return None
                    
                if num_words == 1 and len(final_transcript) < min_chars_single:
                    print(f"[VAD] Rejecting (single short word): '{final_transcript}'")
                    return None

                print(f"[VAD] Accepted final transcript: '{final_transcript}'")
                # Reset transcriber for next utterance
                self.transcriber.reset()
                return final_transcript

            print("[VAD] No final transcript obtained.")
            # Reset transcriber even if no transcript
            self.transcriber.reset()
            return None

        except Exception as e:
            print(f"[ERROR] Error during VAD/transcription: {e}")
            traceback.print_exc()
            # Reset transcriber on error
            self.transcriber.reset()
            return None
            
        finally:
            await self.audio_manager.stop_listening()
            if hasattr(self, "last_partial_print_time"):
                del self.last_partial_print_time

    async def capture_speech_push_to_talk(self, display_manager, immediate_feedback_callback=None) -> str | None:
        """
        Push-to-talk mode: 1 minute capture with no VAD timeouts
        
        KEY DIFFERENCES from VAD mode:
        - User controls timing (key press/release or 60s timeout)
        - No energy threshold detection
        - No silence frame counting
        - VOSK finals are ignored (same as VAD mode)
        - Always accumulates partials for fallback
        - Uses server's final_result when complete
        
        This mode works reliably because it doesn't conflict with VOSK's internal timing
        """
        # Check VOSK readiness before starting speech capture
        if not vosk_readiness.is_speech_enabled():
            print("[WARNING] VOSK server not ready - push-to-talk disabled")
            await display_manager.update_display("error", mood="confused", text="Speech Recognition Unavailable")
            return None
            
        print(f"[PUSH-TO-TALK] Starting extended capture mode (60s max)")
        # Note: audio already initialized and display updated by caller
        audio_stream = await self.audio_manager.start_listening()
        
        if not audio_stream:
            print("[ERROR] Failed to start audio stream for push-to-talk.")
            return None

        # Initialize transcriber
        self.transcriber.reset()
        start_time = time.monotonic()
        max_recording_time = 60.0  # 1 minute max
        callback_triggered = False  # Track if we've triggered the feedback callback
        manual_stop = False  # Track if manually stopped
        last_partial_text = ""  # Track last partial for fallback
        
        try:
            while True:
                current_time = time.monotonic()
                
                # Skip manual stop check during startup delay to prevent detecting initial key press
                if current_time - start_time > 0.5:
                    # Check for manual stop via keyboard (press again to stop)
                    if await self._check_manual_vad_stop():
                        print("[PUSH-TO-TALK] Manual stop via keyboard.")
                        manual_stop = True
                        break
                
                # Max time check
                if current_time - start_time > max_recording_time:
                    print(f"[PUSH-TO-TALK] Max recording time ({max_recording_time}s) reached.")
                    break
                
                # Read and process audio frame
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.005)
                    continue
                
                # Debug: Log that we're sending audio to VOSK (only log first few frames to avoid spam)
                if current_time - start_time < 1.0:
                    print(f"[PUSH-TO-TALK DEBUG] Sending audio frame to VOSK, size: {len(pcm_bytes)} bytes")
                
                # Process with transcriber
                try:
                    is_final_vosk, is_speech, partial_text_vosk = self.transcriber.process_frame(pcm_bytes)
                    
                    # Track last partial text for fallback
                    if partial_text_vosk and len(partial_text_vosk.strip()) > 0:
                        last_partial_text = partial_text_vosk.strip()
                    
                    # Trigger callback on first speech detection (for Claude Code teletype sound)
                    # Use partial text as speech indicator since we're getting consistent partials
                    if partial_text_vosk and len(partial_text_vosk.strip()) > 0 and not callback_triggered:
                        print(f"[PUSH-TO-TALK] First speech detected, triggering callback. Text: '{partial_text_vosk[:50]}...'")
                        if immediate_feedback_callback:
                            print("[PUSH-TO-TALK] Calling immediate_feedback_callback...")
                            await immediate_feedback_callback()
                            print("[PUSH-TO-TALK] Callback completed")
                        else:
                            print("[PUSH-TO-TALK] No callback provided")
                        callback_triggered = True
                    
                    # Show partial results
                    if partial_text_vosk and len(partial_text_vosk.strip()) > 0:
                        if not hasattr(self, "last_partial_print_time") or (current_time - getattr(self, "last_partial_print_time", 0) > 0.5):
                            print(f"[PUSH-TO-TALK] Partial: {partial_text_vosk}")
                            self.last_partial_print_time = current_time
                            
                except Exception as vosk_error:
                    print(f"[PUSH-TO-TALK] Vosk processing error: {vosk_error}")
            
            # Handle manual stop case - give transcriber time to process final audio
            if manual_stop:
                print("[PUSH-TO-TALK] Manual stop detected, allowing transcriber to finalize...")
                # Give a small buffer for any remaining audio processing
                await asyncio.sleep(0.2)
            
            # Get final transcript
            final_transcript = self.transcriber.get_final_text()
            print(f"[PUSH-TO-TALK] Final transcript: '{final_transcript}' (manual_stop: {manual_stop})")
            
            # If final transcript is empty but we have partial text, use the partial as fallback
            if not final_transcript or not final_transcript.strip():
                if last_partial_text:
                    print(f"[PUSH-TO-TALK] Using last partial text as fallback: '{last_partial_text}'")
                    final_transcript = last_partial_text
                else:
                    print("[PUSH-TO-TALK] No final transcript and no partial text available")
                    return None
            
            if final_transcript:
                final_transcript = final_transcript.strip()
                
                # Background noise filter - reject if transcript is just "the" (common noise artifact)
                if final_transcript.lower().strip() == "the":
                    print(f"[PUSH-TO-TALK] Rejecting background noise artifact: '{final_transcript}'")
                    return None
                
                if final_transcript:
                    return final_transcript
                    
            return None
            
        except Exception as e:
            print(f"[ERROR] Error during push-to-talk capture: {e}")
            traceback.print_exc()
            return None
            
        finally:
            await self.audio_manager.stop_listening()
            if hasattr(self, "last_partial_print_time"):
                del self.last_partial_print_time
            # Ensure transcriber state is clean for next capture
            try:
                self.transcriber.reset()
                print("[PUSH-TO-TALK] Transcriber state reset for next capture")
            except Exception as reset_error:
                print(f"[PUSH-TO-TALK] Warning: transcriber reset failed: {reset_error}")