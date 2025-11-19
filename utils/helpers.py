# utils/helpers.py
"""
A collection of standalone helper and utility functions.
"""

import os
import sys
import random
from pathlib import Path

# Keep a backup of the original stderr for restoration
_stderr_backup = sys.stderr
_stderr_fd_backup = os.dup(2)
_null = open(os.devnull, 'w')

SOUND_BASE_PATH = Path("/home/user/more transfer/assets/sounds")

def suppress_jack_errors():
    """Suppress Jack audio and TensorFlow errors at both Python and C++ levels"""
    sys.stderr = _null
    os.dup2(_null.fileno(), 2)

def restore_stderr():
    """Restore stderr after suppression"""
    os.dup2(_stderr_fd_backup, 2)
    sys.stderr = _stderr_backup

def cleanup_old_mood_files():
    """Deletes any leftover duration-encoded TTS files from previous runs."""
    audio_cache_dir = Path("/home/user/claude-to-speech/audio_cache")
    if not audio_cache_dir.exists():
        return
        
    print("[INFO] Cleaning up old mood audio files...")
    try:
        files_to_delete = list(audio_cache_dir.glob("duration_*.mp3"))
        for f in files_to_delete:
            os.remove(f)
        print(f"[INFO] Cleanup complete. Deleted {len(files_to_delete)} file(s).")
    except OSError as e:
        print(f"[ERROR] Error deleting old audio files: {e}")

def get_random_audio(category: str, subtype: str = None):
    """Gets a random audio file for a given category."""
    try:
        base_sound_dir = SOUND_BASE_PATH / "laura"
        
        if category == "wake" and subtype in ["Laura.pmdl", "Wake_up_Laura.pmdl", "GD_Laura.pmdl"]:
            context_map = {
                "Laura.pmdl": "standard", "Wake_up_Laura.pmdl": "sleepy", "GD_Laura.pmdl": "frustrated"
            }
            folder = context_map.get(subtype, "standard")
            audio_path = base_sound_dir / "wake_sentences" / folder
        else:
            audio_path = base_sound_dir / f"{category}_sentences"
            if subtype and (audio_path / subtype).exists():
                audio_path = audio_path / subtype
        
        if audio_path.exists():
            audio_files = list(audio_path.glob('*.mp3')) + list(audio_path.glob('*.wav'))
            if audio_files:
                return str(random.choice(audio_files))
        return None
    except Exception as e:
        print(f"Error in get_random_audio: {e}")
        return None
