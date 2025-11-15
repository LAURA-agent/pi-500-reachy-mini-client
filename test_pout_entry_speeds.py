#!/usr/bin/env python3
"""Test both pout entry speeds: instant (0.5s) vs slow (2.0s)"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove

def test_entry_speeds():
    print("=" * 60)
    print("TEST: Pout Entry Speeds")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    time.sleep(1.5)

    # Test 1: INSTANT entry (0.5s) - comedic snap
    print("\n[TEST 1] INSTANT entry (0.5s) - comedic/reactive")
    print("         Effect: Quick snap into pout, like a reflex")

    pout_instant = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=0.5,  # INSTANT
        target_body_yaw_deg=0.0,
        antenna_twitch_pattern="frustration_twitch"
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_instant)

    print("   → Entering pout NOW")
    time.sleep(1.5)  # 0.5s entry + 1s to observe
    print("   ✓ Instant entry complete - observe the snappy retreat")
    time.sleep(3)

    # Exit to neutral
    print("\n   Returning to neutral...")
    movement_manager.clear_move_queue()
    time.sleep(2)

    # Test 2: SLOW entry (2.0s) - deliberate disdain
    print("\n[TEST 2] SLOW entry (2.0s) - deliberate/disdainful")
    print("         Effect: Methodical controlled retreat, pointed")

    pout_slow = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=2.0,  # SLOW
        target_body_yaw_deg=0.0,
        antenna_twitch_pattern="frustration_twitch"
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_slow)

    print("   → Entering pout NOW")
    time.sleep(3.0)  # 2.0s entry + 1s to observe
    print("   ✓ Slow entry complete - observe the deliberate disdain")
    time.sleep(3)

    print("\n" + "=" * 60)
    print("✅ Test complete - compare the two entry styles")
    print("   INSTANT: Snappy reactive comedy")
    print("   SLOW:    Deliberate controlled disdain")
    print("=" * 60)

    movement_manager.stop()

if __name__ == "__main__":
    test_entry_speeds()
