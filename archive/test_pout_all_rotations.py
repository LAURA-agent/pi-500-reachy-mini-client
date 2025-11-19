#!/usr/bin/env python3
"""Test all pout rotation positions: -90°, -45°, -30°, 0°, 30°, 45°, 90°"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove

def test_all_rotations():
    print("=" * 60)
    print("TEST: All Pout Rotation Positions")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    # Create pout move
    print("\n[SETUP] Creating pout controller...")
    pout_move = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=0.1,
        target_body_yaw_deg=0.0,
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)
    time.sleep(1.5)
    print("   ✓ Pout pose established")

    # Test each allowed rotation position
    positions = [-90, -45, -30, 0, 30, 45, 90]

    for i, angle in enumerate(positions, 1):
        print(f"\n[{i}/{len(positions)}] Rotating to {angle:+4d}° (3s duration)...")
        pout_move.rotate_to_smooth(angle, duration=3.0)

        time.sleep(3.5)  # Wait for rotation + 0.5s settle
        print(f"         ✓ At {angle:+4d}° - observe head/body alignment")
        time.sleep(2)  # Hold position for observation

    # Return to neutral
    print(f"\n[FINAL] Returning to 0° (3s duration)...")
    pout_move.rotate_to_smooth(0, duration=3.0)
    time.sleep(3.5)
    print("         ✓ Back at 0°")

    print("\n" + "=" * 60)
    print("✅ Test complete - all positions tested")
    print("=" * 60)

    movement_manager.stop()

if __name__ == "__main__":
    test_all_rotations()
