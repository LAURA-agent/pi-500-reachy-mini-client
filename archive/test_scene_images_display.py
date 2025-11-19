#!/usr/bin/env python3
"""Test display manager with scene_images."""

import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from display.display_manager import DisplayManager

async def test_display():
    print("=" * 60)
    print("TEST: Scene Images Display Manager")
    print("=" * 60)

    # Initialize display
    print("\n[1/5] Initializing display manager...")
    display = DisplayManager()
    display.initialized = True

    # Debug: Check what was loaded
    print("\n[DEBUG] Image cache keys:", list(display.image_cache.keys())[:10])
    print("[DEBUG] Idle images:", list(display.image_cache.get('idle', {}).keys()) if 'idle' in display.image_cache else "No idle key")

    # Test idle state with various moods
    print("\n[2/5] Testing idle state with different moods...")
    test_moods = ["annoyed", "happy", "confused", "pout", "smug"]

    for mood in test_moods:
        print(f"   Setting state: idle, mood: {mood}")
        display.update_state("idle", mood=mood)
        await asyncio.sleep(2)

    # Test pout state
    print("\n[3/5] Testing pout state...")
    display.update_state("pout")
    await asyncio.sleep(3)

    # Test speaking state
    print("\n[4/5] Testing speaking state with moods...")
    test_speaking_moods = ["happy", "excited", "embarrassed", "worried"]

    for mood in test_speaking_moods:
        print(f"   Setting state: speaking, mood: {mood}")
        display.update_state("speaking", mood=mood)
        await asyncio.sleep(2)

    # Test listening state (symlinked to idle)
    print("\n[5/5] Testing listening state (fallback to idle)...")
    display.update_state("listening", mood="confused")
    await asyncio.sleep(3)

    print("\n" + "=" * 60)
    print("✅ Display test complete!")
    print("   Window should be 640x480 landscape")
    print("   All states and moods should have displayed correctly")
    print("=" * 60)

    # Keep window open for observation
    print("\nWindow will stay open for 10 seconds...")
    await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(test_display())
