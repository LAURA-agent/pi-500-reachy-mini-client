#!/usr/bin/env python3
"""Reorganize scene_images from flat to nested mood structure."""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

SCENE_IMAGES = Path("/home/user/reachy/pi_reachy_deployment/assets/images/laura rp client images/scene_images")

def extract_mood(filename):
    """Extract mood from filename like 'annoyed01.png' -> 'annoyed'"""
    # Remove extension
    name = filename.replace('.png', '')
    # Remove trailing numbers
    mood = re.sub(r'\d+$', '', name)
    return mood

def reorganize_state_folder(state_path):
    """Reorganize a state folder into mood subfolders."""
    print(f"\n[{state_path.name}] Processing...")

    # Group files by mood
    mood_files = defaultdict(list)

    for file in state_path.glob("*.png"):
        mood = extract_mood(file.name)
        mood_files[mood].append(file)

    # Create mood folders and move files
    for mood, files in mood_files.items():
        mood_dir = state_path / mood
        mood_dir.mkdir(exist_ok=True)

        for file in files:
            dest = mood_dir / file.name
            print(f"  {file.name} -> {mood}/{file.name}")
            shutil.move(str(file), str(dest))

    print(f"[{state_path.name}] Created {len(mood_files)} mood folders")

def rename_talking_to_speaking():
    """Rename talking/ folder to speaking/"""
    talking = SCENE_IMAGES / "talking"
    speaking = SCENE_IMAGES / "speaking"

    if talking.exists() and not speaking.exists():
        print(f"\n[RENAME] talking/ -> speaking/")
        talking.rename(speaking)
    elif speaking.exists():
        print(f"\n[SKIP] speaking/ already exists")

def create_fallback_states():
    """Create missing state folders as symlinks to idle."""
    idle_path = SCENE_IMAGES / "idle"
    missing_states = ["listening", "sleep", "thinking", "execution", "wake", "boot", "system", "tool_use", "code", "error", "disconnected"]

    print(f"\n[FALLBACK] Creating missing state folders...")
    for state in missing_states:
        state_path = SCENE_IMAGES / state
        if not state_path.exists():
            # Create as symlink to idle
            state_path.symlink_to("idle")
            print(f"  {state} -> idle (symlink)")

if __name__ == "__main__":
    print("=" * 60)
    print("Reorganizing scene_images to nested mood structure")
    print("=" * 60)

    # 1. Rename talking to speaking first
    rename_talking_to_speaking()

    # 2. Reorganize idle folder
    idle_path = SCENE_IMAGES / "idle"
    if idle_path.exists():
        reorganize_state_folder(idle_path)

    # 3. Reorganize speaking folder (was talking)
    speaking_path = SCENE_IMAGES / "speaking"
    if speaking_path.exists():
        reorganize_state_folder(speaking_path)

    # 4. Pout is already a dedicated state folder, no need to reorganize
    print("\n[pout] Already in correct structure (dedicated state folder)")

    # 5. Create fallback states
    create_fallback_states()

    print("\n" + "=" * 60)
    print("✅ Reorganization complete!")
    print("=" * 60)
