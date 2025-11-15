#!/usr/bin/env python3
"""Debug pout rotation - see what's actually happening."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove

def test_pout_rotation_debug():
    print("=" * 60)
    print("DEBUG: Pout Rotation")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    # Create pout move
    print("\nCreating pout move...")
    pout_move = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=0.1,
        target_body_yaw_deg=0.0,
    )

    print(f"Initial target: {pout_move.target_body_yaw_rad} rad = 0°")

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)
    time.sleep(1)

    # Test single rotation
    print("\n" + "=" * 60)
    print("Testing rotation to 90°")
    print("=" * 60)

    print(f"\nBEFORE rotate_to():")
    print(f"  target_body_yaw_rad = {pout_move.target_body_yaw_rad}")

    pout_move.rotate_to(90)

    print(f"\nAFTER rotate_to(90):")
    print(f"  target_body_yaw_rad = {pout_move.target_body_yaw_rad}")
    print(f"  (should be ~1.57 rad)")

    print("\nWaiting 5 seconds - watch for rotation...")
    for i in range(5):
        time.sleep(1)
        print(f"  {i+1}s - Check: target still = {pout_move.target_body_yaw_rad:.3f} rad")

    # Check if move is still in queue
    print(f"\nMovement manager state:")
    print(f"  current_move = {type(movement_manager.state.current_move).__name__ if movement_manager.state.current_move else 'None'}")
    print(f"  move_queue length = {len(movement_manager.move_queue)}")

    if movement_manager.state.current_move == pout_move:
        print("  ✓ Pout move is active")

        # Manually evaluate the move to see what it returns
        t = time.monotonic() - movement_manager.state.move_start_time if movement_manager.state.move_start_time else 5.0
        head, antennas, body_yaw = pout_move.evaluate(t)
        print(f"\nManual evaluate() at t={t:.1f}s:")
        print(f"  body_yaw returned = {body_yaw:.3f} rad = {body_yaw * 57.3:.1f}°")
    else:
        print("  ✗ Pout move is NOT active!")

    movement_manager.stop()
    print("\nDone!")

if __name__ == "__main__":
    test_pout_rotation_debug()
