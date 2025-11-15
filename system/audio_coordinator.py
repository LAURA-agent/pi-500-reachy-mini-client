#!/usr/bin/env python3

import asyncio
import os
import tempfile
import time
from pathlib import Path
from config.client_config import KEEP_TEMP_AUDIO_FILES


class AudioCoordinator:
    """
    Coordinates audio operations including TTS playback and audio completion waiting.

    Handles TTS audio file management, playback coordination, and proper
    timing for audio device release to prevent feedback loops.
    """

    def __init__(self, audio_manager, speech_analyzer=None, speech_offset_player=None):
        self.audio_manager = audio_manager
        self.speech_analyzer = speech_analyzer
        self.speech_offset_player = speech_offset_player

    async def handle_tts_playback(self, audio_bytes: bytes, source_engine: str, text: str = ""):
        """
        Handle TTS audio playback by writing bytes to file and playing through AudioManager.

        This method writes the TTS audio bytes to a temporary file, analyzes for speech motion,
        and plays with synchronized head movements.

        Args:
            audio_bytes: The audio data from TTS engine
            source_engine: The TTS engine that generated the audio (for file extension)
            text: The text that was synthesized (for speech analysis)
        """
        if not audio_bytes:
            print("[AudioCoordinator.handle_tts_playback] No audio bytes to play.")
            return

        temp_dir = Path(tempfile.gettempdir())
        # Use .wav for Piper as it produces WAV or similar PCM output, MP3 for ElevenLabs/Cartesia.
        ext = ".wav" if source_engine.lower() in ["piper"] else ".mp3"
        fname = temp_dir / f"assistant_response_{int(time.time()*1000)}{ext}"

        try:
            # Write the TTS audio bytes to temporary file
            with open(fname, "wb") as f:
                f.write(audio_bytes)
            print(f"[AudioCoordinator.handle_tts_playback] Audio file written: {fname} ({len(audio_bytes)} bytes)")

            # Analyze audio for speech motion if analyzer is available
            if self.speech_analyzer and self.speech_offset_player and text:
                print(f"[AudioCoordinator] Analyzing audio for speech motion...")
                loop = asyncio.get_event_loop()
                analysis = await loop.run_in_executor(None, self.speech_analyzer.analyze, str(fname), text)
                self.speech_offset_player.load_timeline(analysis)
                print(f"[AudioCoordinator] Speech motion timeline loaded")

            # Create audio playback task but don't await it yet
            audio_playback_task = asyncio.create_task(self.audio_manager.play_audio(str(fname)))

            # Capture audio start time and start speech motion BEFORE audio blocks
            audio_start_time = time.time()

            # Start speech motion playback if available
            if self.speech_offset_player and text:
                self.speech_offset_player.play(audio_start_time)
                print(f"[AudioCoordinator] Speech motion playback started at t={audio_start_time:.3f}")

            # Now wait for audio playback to complete
            await audio_playback_task
            print(f"[AudioCoordinator.handle_tts_playback] Audio playback completed for: {fname}")

            # Stop speech motion
            if self.speech_offset_player and text:
                self.speech_offset_player.stop()
                print(f"[AudioCoordinator] Speech motion playback stopped")

        except Exception as e:
            print(f"[ERROR] AudioCoordinator.handle_tts_playback: Failed to play audio from {fname}: {e}")
            # Stop speech motion on error
            if self.speech_offset_player:
                self.speech_offset_player.stop()
        finally:
            # Clean up temporary file after playback is confirmed complete
            if os.path.exists(fname) and not KEEP_TEMP_AUDIO_FILES:
                try:
                    os.remove(fname)
                    print(f"[AudioCoordinator.handle_tts_playback] Temp audio file deleted: {fname}")
                except Exception as e_del:
                    print(f"[WARN] Failed to delete temp audio file {fname}: {e_del}")

    async def play_audio_file(self, audio_file_path: str):
        """Play an audio file directly through the audio manager"""
        await self.audio_manager.play_audio(audio_file_path)

    async def wait_for_audio_completion_with_buffer(self):
        """Wait for audio completion with additional buffer time to ensure device release"""
        # First wait for the audio manager's completion event
        await self.audio_manager.wait_for_audio_completion()
        
        # Add a small fixed buffer to ensure audio device is fully released
        buffer_time = 0.5  # Half second buffer regardless of audio length
        await asyncio.sleep(buffer_time)

    async def stop_current_audio(self):
        """Stop any currently playing audio"""
        await self.audio_manager.stop_current_audio()

    async def wait_for_audio_completion(self):
        """Standard audio completion wait without additional buffer"""
        await self.audio_manager.wait_for_audio_completion()

    async def cleanup(self):
        """Clean up audio coordinator resources"""
        # Audio manager cleanup is handled by the audio manager itself
        pass