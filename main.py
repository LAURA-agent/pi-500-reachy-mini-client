# main.py
"""
Main entry point for the Pi Reachy application.
Initializes the system, handles startup checks, and runs the main orchestrator.
"""
import asyncio
import os
import sys
import platform
from pathlib import Path

# Suppress initial warnings before any other imports
os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
os.environ['JACK_NO_START_SERVER'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_VMODULE'] = 'inference_feedback_manager=0'

# Add snowboy to path for wake word detection (Linux only)
if platform.system() != "Darwin":
    sys.path.insert(0, str(Path(__file__).parent / "snowboy"))

from core.app_orchestrator import AppOrchestrator
from utils.helpers import cleanup_old_mood_files, restore_stderr

async def main():
    """Main entry point with multi-task architecture."""
    print("--- STARTING PI REACHY APPLICATION ---", flush=True)

    # Clean up old mood files before starting
    cleanup_old_mood_files()

    # Check speech server readiness (Linux only)
    if platform.system() != "Darwin":
        from speech_capture.vosk_readiness_checker import ensure_vosk_ready
        print("[INFO] Checking speech server readiness...")
        if not await ensure_vosk_ready(timeout=10):
            print("[WARNING] Speech server not ready - features will be limited")
        else:
            print("[INFO] Speech server ready")
    else:
        print("[INFO] Running in Passive Mode - Hardware Disabled")

    print("[INFO] Initializing Application Orchestrator...")
    orchestrator = AppOrchestrator()
    
    # Restore stderr after all initializations are complete
    restore_stderr()
    print("[INFO] Client initialized successfully")

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Shutting down...")
    finally:
        print("[INFO] Main function finished. Performing final cleanup...")
        await orchestrator.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Application terminated by user.")
    finally:
        print("[INFO] Application shutdown complete.")
