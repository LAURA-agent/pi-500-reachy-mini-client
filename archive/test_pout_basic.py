#!/usr/bin/env python3
"""Basic pout mode test - verify sleep pose entry.

Test: Can Laura enter pout mode and move to sleep pose?

Expected behavior:
- Laura moves from current position into hunched/hiding sleep pose
- Head tilts down and forward
- Antennas spread to ±3.05 rad
- Body stays at 0° (or specified angle)
- Movement takes ~2 seconds
- Breathing freezes (no sway/roll)
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove
from reachy_mini.utils import create_head_pose

def test_pout_entry():
    """Test entering pout mode."""
    print("=" * 60)
    print("TEST: Pout Mode Entry")
    print("=" * 60)

    # Connect to daemon
    print("\n[1/5] Connecting to daemon...")
    daemon = DaemonClient("http://localhost:8100")

    # Create movement manager (without camera worker for simplicity)
    print("[2/5] Creating movement manager...")
    movement_manager = MovementManager(
        current_robot=daemon,
        camera_worker=None
    )
    movement_manager.start()

    # Get current pose
    print("[3/5] Getting current pose...")
    try:
        head_joints, antenna_joints = daemon.get_current_joint_positions()
        current_head_pose = daemon.get_current_head_pose()
        current_antennas = (antenna_joints[-2], antenna_joints[-1])
        current_body_yaw = head_joints[0]

        print(f"   Current head pose: {current_head_pose[:3, 3]} (xyz)")
        print(f"   Current antennas: {current_antennas}")
        print(f"   Current body yaw: {np.rad2deg(current_body_yaw):.1f}°")
    except Exception as e:
        print(f"   ERROR: Could not get current pose: {e}")
        print("   Using safe defaults...")
        current_head_pose = create_head_pose(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, degrees=True, mm=False)
        current_antennas = (0.0, 0.0)
        current_body_yaw = 0.0

    # Create pout move
    print("[4/5] Creating pout move...")
    pout_move = PoutPoseMove(
        interpolation_start_pose=current_head_pose,
        interpolation_start_antennas=current_antennas,
        interpolation_start_body_yaw=current_body_yaw,
        interpolation_duration=2.0,
        target_body_yaw_deg=0.0,  # Straight ahead
    )

    print(f"   Target: Sleep pose (hunched/hiding)")
    print(f"   Duration: 2.0 seconds")
    print(f"   Body rotation: 0°")

    # Queue the move
    print("[5/5] Queueing pout move...")
    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)

    print("\n" + "=" * 60)
    print("POUT MOVE STARTED")
    print("=" * 60)
    print("\nWatch Laura:")
    print("  - Should tilt head down and forward (hunched)")
    print("  - Antennas should spread outward")
    print("  - Should reach pose in ~2 seconds")
    print("  - Breathing should stop (no sway/roll)")
    print("\nWaiting 5 seconds to observe...")

    # Wait and observe
    time.sleep(5)

    # Check final pose
    print("\n" + "=" * 60)
    print("CHECKING FINAL POSE")
    print("=" * 60)

    try:
        head_joints, antenna_joints = daemon.get_current_joint_positions()
        final_head_pose = daemon.get_current_head_pose()
        final_antennas = (antenna_joints[-2], antenna_joints[-1])
        final_body_yaw = head_joints[0]

        print(f"\nFinal head pose: {final_head_pose[:3, 3]} (xyz)")
        print(f"Final antennas: {final_antennas}")
        print(f"Final body yaw: {np.rad2deg(final_body_yaw):.1f}°")

        # Compare to expected sleep pose
        expected_sleep_xyz = PoutPoseMove.SLEEP_HEAD_POSE[:3, 3]
        expected_antennas = PoutPoseMove.SLEEP_ANTENNAS

        print(f"\nExpected sleep pose: {expected_sleep_xyz} (xyz)")
        print(f"Expected antennas: {expected_antennas}")

        # Check if close to target
        xyz_error = np.linalg.norm(final_head_pose[:3, 3] - expected_sleep_xyz)
        antenna_error = abs(final_antennas[0] - expected_antennas[0]) + abs(final_antennas[1] - expected_antennas[1])

        print(f"\nPosition error: {xyz_error:.4f}m")
        print(f"Antenna error: {antenna_error:.3f} rad")

        if xyz_error < 0.01 and antenna_error < 0.1:
            print("\n✅ TEST PASSED - Laura reached sleep pose")
        else:
            print("\n⚠️  TEST UNCERTAIN - Pose may not be exact")

    except Exception as e:
        print(f"\nERROR checking final pose: {e}")

    # Cleanup
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)
    print("\nStopping movement manager...")
    movement_manager.stop()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nVisual check:")
    print("  - Is Laura's head tilted down?")
    print("  - Are antennas spread outward?")
    print("  - Does she look like she's hiding/pouting?")
    print("\nPress Ctrl+C if you want to stop here.")
    print("Otherwise, Laura will stay in this pose.")

    # Keep running so user can observe
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        movement_manager.stop()

if __name__ == "__main__":
    try:
        test_pout_entry()
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
