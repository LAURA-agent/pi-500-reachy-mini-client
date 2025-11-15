#!/usr/bin/env python3
"""Test single pout rotation - 0° → 30° → 0°"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove

def test_single_rotation():
    print("=" * 60)
    print("TEST: Single Pout Rotation (0° → 30° → 0°)")
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

    # Rotate to 30°
    print("\n[1/2] Rotating to +30°...")
    pout_move.rotate_to(30)
    time.sleep(4)
    print("   Should be at 30° now")

    # Return to 0°
    print("\n[2/2] Rotating back to 0°...")
    pout_move.rotate_to(0)
    time.sleep(4)
    print("   Should be at 0° now")

    print("\n✅ Test complete")
    movement_manager.stop()

if __name__ == "__main__":
    test_single_rotation()
