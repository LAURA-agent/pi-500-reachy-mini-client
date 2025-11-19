#!/usr/bin/env python3
"""
Reachy Mini Mood Hook - Pi Version with move duration timing
Uses actual RecordedMove durations instead of fixed 3-5s waits
"""

import re
import sys
import time
import random
import requests
import json
import glob
from datetime import datetime
from pathlib import Path

# Reachy Mini imports for move duration
try:
    from reachy_mini.motion.recorded_move import RecordedMove
    HAVE_REACHY_SDK = True
except ImportError:
    HAVE_REACHY_SDK = False
    print("[WARNING] reachy_mini SDK not available, using fallback timing", file=sys.stderr)

# Daemon configuration
DAEMON_URL = "http://localhost:8100"
DATASET = "pollen-robotics/reachy-mini-emotions-library"

# TTS server status endpoint
TTS_STATUS_URL = "http://localhost:5001/status"

# EXISTING conversation app endpoints (not arbiter)
EXTERNAL_START_URL = "http://localhost:7860/api/external_control/start"
EXTERNAL_STOP_URL = "http://localhost:7860/api/external_control/stop"
EXTERNAL_STATUS_URL = "http://localhost:7860/api/external_control/status"

# Mood categories with mapped emotions
MOOD_CATEGORIES = {
    "celebratory": [
        "success1", "success2", "proud1", "proud2", "proud3",
        "cheerful1", "electric1", "enthusiastic1", "enthusiastic2"
    ],
    "thoughtful": [
        "thoughtful1", "thoughtful2", "curious1", "attentive1", "attentive2",
        "inquiring1", "inquiring2", "inquiring3"
    ],
    "welcoming": [
        "welcoming1", "welcoming2", "helpful1", "helpful2", "loving1",
        "come1", "grateful1", "cheerful1"
    ],
    "confused": [
        "confused1", "uncertain1", "lost1", "inquiring1", "inquiring2",
        "incomprehensible2", "uncomfortable1"
    ],
    "frustrated": [
        "frustrated1", "irritated1", "irritated2", "impatient1", "impatient2",
        "exhausted1", "tired1"
    ],
    "surprised": [
        "surprised1", "surprised2", "amazed1", "oops1", "oops2",
        "incomprehensible2", "electric1"
    ],
    "calm": [
        "calming1", "serenity1", "relief1", "relief2", "shy1",
        "understanding1", "understanding2"
    ],
    "energetic": [
        "electric1", "enthusiastic1", "enthusiastic2", "dance1", "dance2",
        "dance3", "laughing1", "laughing2"
    ],
    "playful": [
        "laughing1", "laughing2", "dance1", "dance2", "dance3",
        "cheerful1", "enthusiastic1", "oops1", "oops2"
    ]
}


def log_debug(msg):
    """Print timestamped debug message to stderr."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [MOOD DEBUG] {msg}", file=sys.stderr)


def extract_mood_marker(text):
    """Extract <!-- MOOD: mood_name --> marker from text."""
    pattern = r'<!--\s*MOOD:\s*([a-zA-Z0-9_]+)\s*-->'
    match = re.search(pattern, text)
    result = match.group(1) if match else None
    log_debug(f"Extracted mood marker: {result}")
    return result


def notify_external_control_start():
    """Notify conversation app that we're taking control."""
    log_debug("Attempting to start external control...")
    try:
        response = requests.post(EXTERNAL_START_URL, timeout=2)
        log_debug(f"External control start response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            log_debug(f"External control response data: {data}")
            return True
        else:
            log_debug(f"External control start failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        log_debug(f"External control start error: {e}")
        return False


def notify_external_control_stop():
    """Notify conversation app that we're done."""
    log_debug("Attempting to stop external control...")
    try:
        response = requests.post(EXTERNAL_STOP_URL, timeout=2)
        log_debug(f"External control stop response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            log_debug(f"External control stop data: {data}")
            return True
        else:
            log_debug(f"External control stop failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        log_debug(f"External control stop error: {e}")
        return False


def check_external_control_status():
    """Check if external control is active."""
    try:
        response = requests.get(EXTERNAL_STATUS_URL, timeout=1)
        if response.status_code == 200:
            data = response.json()
            is_active = data.get('external_control_active', False)
            log_debug(f"External control status check: active={is_active}")
            return is_active
        else:
            log_debug(f"External control status failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        log_debug(f"External control status error: {e}")
        return False


def is_tts_playing():
    """Check if TTS is currently playing."""
    try:
        response = requests.get(TTS_STATUS_URL, timeout=1)
        if response.status_code == 200:
            data = response.json()
            playing = data.get('is_playing', False)
            log_debug(f"TTS status: is_playing={playing}")
            return playing
        else:
            log_debug(f"TTS status check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        log_debug(f"TTS status check error: {e}")
        return False


def get_running_moves():
    """Get list of currently running moves from daemon."""
    url = f"{DAEMON_URL}/api/move/running"
    log_debug(f"Querying running moves: {url}")

    try:
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            data = response.json()
            move_uuids = [item['uuid'] for item in data]
            log_debug(f"Found {len(move_uuids)} running moves: {move_uuids}")
            return move_uuids
        else:
            log_debug(f"Failed to get running moves: HTTP {response.status_code}")
            return []
    except Exception as e:
        log_debug(f"Error getting running moves: {e}")
        return []


def stop_move(move_uuid):
    """Stop a specific move by UUID."""
    url = f"{DAEMON_URL}/api/move/stop"
    log_debug(f"Stopping move: {move_uuid}")

    try:
        response = requests.post(url, json={"uuid": move_uuid}, timeout=1)
        if response.status_code == 200:
            log_debug(f"Successfully stopped move: {move_uuid}")
            return True
        else:
            log_debug(f"Failed to stop move {move_uuid}: HTTP {response.status_code}")
            return False
    except Exception as e:
        log_debug(f"Error stopping move {move_uuid}: {e}")
        return False


def get_move_duration(emotion_name):
    """Get the duration of a move from the dataset.

    Returns:
        float: Duration in seconds, or 4.0 as fallback
    """
    if not HAVE_REACHY_SDK:
        return 4.0  # Fallback duration

    try:
        move = RecordedMove.from_dataset_and_name(DATASET, emotion_name)
        duration = move.duration
        log_debug(f"Move '{emotion_name}' duration: {duration:.2f}s")
        return duration
    except Exception as e:
        log_debug(f"Could not get duration for '{emotion_name}': {e}, using 4.0s fallback")
        return 4.0


def play_emotion(emotion_name):
    """Play a single emotion move via daemon API. Returns (success, uuid, duration)."""
    url = f"{DAEMON_URL}/api/move/play/recorded-move-dataset/{DATASET}/{emotion_name}"
    log_debug(f"Playing emotion: {emotion_name}")
    log_debug(f"Request URL: {url}")

    # Get move duration BEFORE playing (so we know how long to wait)
    duration = get_move_duration(emotion_name)

    try:
        response = requests.post(url, timeout=2)
        log_debug(f"Emotion play response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            move_uuid = data.get("uuid", "unknown")
            log_debug(f"Emotion started successfully: {emotion_name} (UUID: {move_uuid}, duration: {duration:.2f}s)")
            return True, move_uuid, duration
        else:
            log_debug(f"Failed to play emotion: HTTP {response.status_code}")
            if response.text:
                log_debug(f"Error response: {response.text}")
            return False, None, 0.0
    except Exception as e:
        log_debug(f"Error playing emotion {emotion_name}: {e}")
        return False, None, 0.0


def wait_for_tts_file(timeout=5.0):
    """Wait for TTS audio file to be created and return its path and duration.

    Parses duration from filename format: duration_{ms}_tts_{timestamp}_{uuid}.mp3

    Returns:
        tuple: (file_path, duration_seconds) or (None, 0) if timeout
    """
    tts_output_dir = Path("/home/user/claude-to-speech/audio_cache")
    log_debug(f"Waiting for TTS file in {tts_output_dir}...")

    start_time = time.time()
    last_file = None

    while time.time() - start_time < timeout:
        if tts_output_dir.exists():
            # Find newest .mp3 file with duration_ prefix
            mp3_files = list(tts_output_dir.glob("duration_*.mp3"))
            if mp3_files:
                newest = max(mp3_files, key=lambda p: p.stat().st_mtime)

                # Check if it's new (created within last 10 seconds)
                age = time.time() - newest.stat().st_mtime
                if age < 10.0:
                    # Wait a moment for file to be fully written
                    time.sleep(0.2)

                    # Parse duration from filename
                    try:
                        # Filename format: duration_{ms}_tts_{timestamp}_{uuid}.mp3
                        filename = newest.name
                        if filename.startswith("duration_"):
                            duration_str = filename.split("_")[1]
                            duration_ms = int(duration_str)
                            duration_seconds = duration_ms / 1000.0
                            log_debug(f"Found TTS file: {filename}, duration: {duration_seconds:.2f}s")
                            return (str(newest), duration_seconds)
                        else:
                            log_debug(f"File doesn't match expected format: {filename}")
                    except (IndexError, ValueError) as e:
                        log_debug(f"Error parsing duration from filename: {e}")

        time.sleep(0.1)

    log_debug(f"No TTS file found within {timeout}s")
    return (None, 0)


def play_mood_loop_debug(mood_name, max_duration=60):
    """Play emotion moves synchronized with TTS audio duration.

    Algorithm:
    1. START IMMEDIATELY - breathing already stopped
    2. Play first 2 random moves right away
    3. While playing, get TTS duration asynchronously
    4. Calculate remaining time after those 2 moves
    5. Fill remaining time with more moves (if they fit)
    6. Sleep any leftover time
    7. Exit cleanly (breathing resumes automatically)
    """
    log_debug(f"=" * 60)
    log_debug(f"STARTING MOOD LOOP: {mood_name}")
    log_debug(f"=" * 60)

    # Validate mood - NO FALLBACK, just exit if invalid
    if mood_name not in MOOD_CATEGORIES:
        log_debug(f"Invalid mood '{mood_name}', exiting cleanly")
        return

    emotions = MOOD_CATEGORIES[mood_name]
    log_debug(f"Available emotions for {mood_name}: {emotions}")

    start_time = time.time()
    moves_played = 0
    cumulative_duration = 0.0
    buffer_per_move = 0.1  # Small buffer between moves

    try:
        # PHASE 1: Play first 2 moves IMMEDIATELY (don't wait for TTS)
        log_debug("\n=== PHASE 1: Playing initial 2 moves ===")

        for initial_move_num in [1, 2]:
            emotion = random.choice(emotions)
            log_debug(f"\n--- Initial move {initial_move_num}: {emotion} ---")

            success, move_uuid, actual_duration = play_emotion(emotion)
            if success:
                moves_played += 1
                wait_time = actual_duration + buffer_per_move
                cumulative_duration += wait_time
                log_debug(f"Move playing, waiting {wait_time:.2f}s (cumulative: {cumulative_duration:.2f}s)...")
                time.sleep(wait_time)
            else:
                log_debug(f"Failed to play {emotion}, continuing anyway")

        # PHASE 2: Get TTS duration (should be available by now)
        log_debug("\n=== PHASE 2: Getting TTS duration ===")
        tts_file, tts_duration = wait_for_tts_file(timeout=10.0)

        if tts_file is None or tts_duration == 0:
            log_debug("No TTS detected, exiting after initial moves")
            return

        log_debug(f"TTS duration: {tts_duration:.2f}s")
        log_debug(f"Already played: {cumulative_duration:.2f}s")

        remaining_duration = tts_duration - cumulative_duration
        log_debug(f"Remaining duration to fill: {remaining_duration:.2f}s")

        # PHASE 3: Fill remaining time with more moves
        log_debug("\n=== PHASE 3: Filling remaining time ===")

        while remaining_duration > 0:
            emotion = random.choice(emotions)
            move_duration = get_move_duration(emotion)
            needed_time = move_duration + buffer_per_move

            if needed_time <= remaining_duration:
                # Move fits within remaining time, play it
                log_debug(f"\n--- Move {moves_played + 1}: {emotion} ({move_duration:.2f}s) fits ---")
                success, move_uuid, actual_duration = play_emotion(emotion)

                if success:
                    moves_played += 1
                    wait_time = actual_duration + buffer_per_move
                    cumulative_duration += wait_time
                    remaining_duration -= wait_time
                    log_debug(f"Move playing, waiting {wait_time:.2f}s (remaining: {remaining_duration:.2f}s)...")
                    time.sleep(wait_time)
                else:
                    log_debug(f"Failed to play {emotion}, stopping")
                    break
            else:
                # Move exceeds remaining time - play it as final move and exit
                log_debug(f"\n--- FINAL Move {moves_played + 1}: {emotion} ({move_duration:.2f}s) exceeds limit, playing anyway ---")
                success, move_uuid, actual_duration = play_emotion(emotion)

                if success:
                    moves_played += 1
                    wait_time = actual_duration + buffer_per_move
                    log_debug(f"Final move playing, waiting {wait_time:.2f}s then exiting to resume breathing...")
                    time.sleep(wait_time)

                # Exit after final move completes
                break

    except KeyboardInterrupt:
        log_debug("Interrupted by user")
    except Exception as e:
        log_debug(f"Unexpected error in mood loop: {e}")
        import traceback
        log_debug(traceback.format_exc())
    finally:
        elapsed = time.time() - start_time
        log_debug("\n--- Cleanup ---")
        log_debug(f"Total moves played: {moves_played}")
        log_debug(f"Total elapsed: {elapsed:.1f}s")
        log_debug("Mood subprocess exiting cleanly, breathing will resume")

    log_debug(f"=" * 60)
    log_debug(f"MOOD LOOP COMPLETE")
    log_debug(f"=" * 60)


def main():
    """Main entry point."""
    log_debug("Mood extractor script started")

    # Read stdin
    text = sys.stdin.read()
    log_debug(f"Read {len(text)} characters from stdin")

    # Extract mood marker
    mood = extract_mood_marker(text)

    if not mood:
        log_debug("No mood marker found, exiting silently")
        sys.exit(0)

    # Run the mood loop
    play_mood_loop_debug(mood)

    log_debug("Script complete")
    sys.exit(0)


if __name__ == "__main__":
    main()