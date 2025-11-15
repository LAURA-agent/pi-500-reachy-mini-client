#!/usr/bin/env python3
"""Test smooth pout rotation - 0° → 30° → 0° with 3s interpolation"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove

def test_smooth_rotation():
    print("=" * 60)
    print("TEST: Smooth Pout Rotation (0° → 30° → 0°)")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    # Create pout move (assumes already in pout pose)
    print("\nCreating pout controller...")
    pout_move = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=0.1,
        target_body_yaw_deg=0.0,
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)
    time.sleep(1)

    # Smoothly rotate to 30° over 3 seconds
    print("\n[1/2] Smoothly rotating to +30° (3s duration)...")
    pout_move.rotate_to_smooth(30, duration=3.0)
    print("   Interpolating...")
    time.sleep(4)  # Wait for 3s interpolation + 1s to settle
    print("   Should be at 30° now")

    # Smoothly return to 0° over 3 seconds
    print("\n[2/2] Smoothly rotating back to 0° (3s duration)...")
    pout_move.rotate_to_smooth(0, duration=3.0)
    print("   Interpolating...")
    time.sleep(4)  # Wait for 3s interpolation + 1s to settle
    print("   Should be at 0° now")

    print("\n✅ Test complete - verify movements were smooth and actuators unstressed")
    movement_manager.stop()

if __name__ == "__main__":
    test_smooth_rotation()
