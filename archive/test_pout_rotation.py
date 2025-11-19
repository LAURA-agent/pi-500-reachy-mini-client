#!/usr/bin/env python3
"""Test pout mode body rotation - verify discrete angle turns.

Test: Can Laura rotate her body while maintaining the pout pose?

Expected behavior:
- Laura enters pout mode facing forward (0°)
- Rotates through discrete angles: 0° → 45° → 90° → -90° → -45° → 0°
- Head stays in sleep pose throughout
- Each rotation takes ~2 seconds to reach target
- Antennas stay spread outward
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

def test_pout_rotation():
    """Test body rotation while pouting."""
    print("=" * 60)
    print("TEST: Pout Mode Body Rotation")
    print("=" * 60)

    # Connect to daemon
    print("\n[1/4] Connecting to daemon...")
    daemon = DaemonClient("http://localhost:8100")

    # Create movement manager
    print("[2/4] Creating movement manager...")
    movement_manager = MovementManager(
        current_robot=daemon,
        camera_worker=None
    )
    movement_manager.start()

    # Get current pose
    print("[3/4] Getting current pose...")
    try:
        head_joints, antenna_joints = daemon.get_current_joint_positions()
        current_head_pose = daemon.get_current_head_pose()
        current_antennas = (antenna_joints[-2], antenna_joints[-1])
        current_body_yaw = head_joints[0]
    except Exception as e:
        print(f"   ERROR: {e}")
        current_head_pose = create_head_pose(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, degrees=True, mm=False)
        current_antennas = (0.0, 0.0)
        current_body_yaw = 0.0

    # Enter pout mode at 0°
    print("[4/4] Entering pout mode at 0°...")
    pout_move = PoutPoseMove(
        interpolation_start_pose=current_head_pose,
        interpolation_start_antennas=current_antennas,
        interpolation_start_body_yaw=current_body_yaw,
        interpolation_duration=2.0,
        target_body_yaw_deg=0.0,
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)

    print("\nWaiting for pout entry (2 seconds)...")
    time.sleep(3)

    print("\n" + "=" * 60)
    print("ROTATION SEQUENCE")
    print("=" * 60)

    # Test rotation sequence
    angles = [0, 45, 90, -90, -45, 0]

    for i, angle in enumerate(angles, 1):
        print(f"\n[{i}/{len(angles)}] Rotating to {angle:+4d}°")

        # Update pout move target
        pout_move.rotate_to(angle)

        # Wait for rotation
        print(f"   Waiting 3 seconds to reach target...")
        time.sleep(3)

        # Check current body yaw
        try:
            head_joints, _ = daemon.get_current_joint_positions()
            current_yaw = head_joints[0]
            current_yaw_deg = np.rad2deg(current_yaw)
            error = abs(current_yaw_deg - angle)

            print(f"   Current: {current_yaw_deg:+6.1f}° (error: {error:.1f}°)")

            if error < 5.0:
                print(f"   ✅ Reached target")
            else:
                print(f"   ⚠️  May not have reached target")

        except Exception as e:
            print(f"   ERROR checking position: {e}")

    print("\n" + "=" * 60)
    print("ROTATION TEST COMPLETE")
    print("=" * 60)

    print("\nVisual verification:")
    print("  - Did Laura rotate through all angles?")
    print("  - Did head stay in pout pose (not lifting up)?")
    print("  - Did antennas stay spread throughout?")
    print("  - Were rotations smooth?")

    print("\n✅ Test finished successfully")
    print("\nPress Ctrl+C to stop, or wait for auto-cleanup in 5 seconds...")

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Cleanup
    print("\nStopping movement manager...")
    movement_manager.stop()
    print("Done!")

if __name__ == "__main__":
    try:
        test_pout_rotation()
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
