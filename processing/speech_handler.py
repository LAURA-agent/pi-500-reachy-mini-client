# processing/speech_handler.py

from speech_capture.vosk_websocket_adapter import VoskTranscriber
from speech_capture.speech_processor import SpeechProcessor
from config.client_config import AUDIO_SAMPLE_RATE

class SpeechHandler:
    """
    Manages speech capture and transcription using VOSK.
    """
    def __init__(self, audio_manager, state_tracker):
        self.audio_manager = audio_manager
        self.state_tracker = state_tracker
        self.transcriber = VoskTranscriber(sample_rate=AUDIO_SAMPLE_RATE)
        self.speech_processor = SpeechProcessor(
            self.audio_manager,
            self.transcriber,
            None, # keyboard_device, set later if needed
            self.audio_manager.vad_queue
        )

    async def capture_speech(self, wake_event_source: str) -> str | None:
        """
        Captures and transcribes speech based on the wake event source.
        
        Returns:
            The transcribed text, or None if no speech was detected.
        """
        await self.audio_manager.initialize_input()

        if "keyboard" in wake_event_source:
            print("[INFO] Using push-to-talk mode (no VAD timeouts)")
            transcript = await self.speech_processor.capture_speech_push_to_talk(self.state_tracker)
        else:
            print("[INFO] Using VAD mode with standard timeouts")
            transcript = await self.speech_processor.capture_speech_with_unified_vad(self.state_tracker)

        return transcript