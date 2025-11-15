#!/usr/bin/env python3
"""Test pout pose WITH integrated antenna twitches"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove

def test_pout_with_twitches():
    print("=" * 60)
    print("TEST: Pout Pose + Antenna Twitches")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    # Enter pout pose WITH antenna twitching enabled
    print("\n[SETUP] Entering pout pose with antenna twitches...")
    pout_move = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=2.0,
        target_body_yaw_deg=0.0,
        antenna_twitch_pattern="frustration_twitch"  # Enable twitching!
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)
    time.sleep(2.5)
    print("   ✓ Pout pose established with twitching enabled")

    # Observe twitching at 0°
    print("\n[TEST 1] Observe antenna twitches at 0° for 15 seconds...")
    for i in range(15):
        time.sleep(1)
        print(f"      {i+1}s", end="\r")

    print("\n   ✓ Twitches at 0° observed")

    # Rotate to +45° while twitching
    print("\n[TEST 2] Rotating to +45° while twitching continues...")
    pout_move.rotate_to_smooth(45, duration=3.0)
    time.sleep(8)  # 3s rotation + 5s observation
    print("   ✓ Twitches at +45° observed")

    # Return to 0°
    print("\n[TEST 3] Returning to 0° while twitching continues...")
    pout_move.rotate_to_smooth(0, duration=3.0)
    time.sleep(5)
    print("   ✓ Twitches at 0° observed")

    print("\n" + "=" * 60)
    print("✅ Test complete - pout + twitches working together")
    print("=" * 60)

    movement_manager.stop()

if __name__ == "__main__":
    test_pout_with_twitches()
