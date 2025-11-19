#!/usr/bin/env python3
"""Test pout antenna twitch patterns"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove, AntennaTwitchMove

def test_antenna_twitch():
    print("=" * 60)
    print("TEST: Pout Antenna Twitch Patterns")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    # Enter pout pose at 0°
    print("\n[SETUP] Entering pout pose...")
    pout_move = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=2.0,
        target_body_yaw_deg=0.0,
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)
    time.sleep(2.5)
    print("   ✓ Pout pose established")

    # Start antenna twitch
    print("\n[TEST] Starting frustration twitch pattern...")
    print("        Pattern: Random flicks at ±3° every 1.0-1.5 seconds")
    print("        Base position: Sleep antennas (-3.05, +3.05)")

    antenna_twitch = AntennaTwitchMove(
        pattern="frustration_twitch",
        base_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS)
    )

    movement_manager.queue_move(antenna_twitch)

    print("\n   Observe antenna twitches for 20 seconds...")
    for i in range(20):
        time.sleep(1)
        print(f"      {i+1}s...", end="\r")

    print("\n\n   ✓ Pattern observation complete")

    # Test with body rotation
    print("\n[TEST] Testing antenna twitch while rotating body...")
    print("        Rotating to +45° while antennas continue twitching")
    pout_move.rotate_to_smooth(45, duration=3.0)
    time.sleep(5)

    print("        Rotating to -45° while antennas continue twitching")
    pout_move.rotate_to_smooth(-45, duration=3.0)
    time.sleep(5)

    print("        Returning to 0°")
    pout_move.rotate_to_smooth(0, duration=3.0)
    time.sleep(3.5)

    print("   ✓ Body rotation with antennas tested")

    print("\n" + "=" * 60)
    print("✅ Test complete - antenna patterns verified")
    print("=" * 60)

    movement_manager.stop()

if __name__ == "__main__":
    test_antenna_twitch()
