#!/usr/bin/env python3
"""Test pout mode body rotation - simplified version.

PREREQUISITE: Laura must already be in pout/sleep pose!
Run test_pout_basic.py first to enter pout mode.

Test: Can Laura rotate her body while maintaining the pout pose?

Expected behavior:
- Assumes Laura is already in sleep pose
- Rotates through discrete angles without moving head out of pose
- Each rotation shows visual change
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, PoutPoseMove

def test_pout_rotation_simple():
    """Test body rotation while already in pout pose."""
    print("=" * 60)
    print("TEST: Pout Mode Body Rotation (Simplified)")
    print("=" * 60)
    print("\nPREREQUISITE: Laura should already be in pout/sleep pose!")
    print("If not, run test_pout_basic.py first.\n")

    input("Press Enter when Laura is in pout pose and ready...")

    # Connect to daemon
    print("\n[1/3] Connecting to daemon...")
    daemon = DaemonClient("http://localhost:8100")

    # Create movement manager
    print("[2/3] Creating movement manager...")
    movement_manager = MovementManager(
        current_robot=daemon,
        camera_worker=None
    )
    movement_manager.start()

    # Create pout move (assumes already at sleep pose, so no interpolation needed)
    print("[3/3] Creating pout move controller...")

    # Use sleep pose as both start and target
    # This makes interpolation instant since start == target
    pout_move = PoutPoseMove(
        interpolation_start_pose=PoutPoseMove.SLEEP_HEAD_POSE.copy(),
        interpolation_start_antennas=tuple(PoutPoseMove.SLEEP_ANTENNAS),
        interpolation_start_body_yaw=0.0,
        interpolation_duration=0.1,  # Very short since we're already there
        target_body_yaw_deg=0.0,
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(pout_move)

    print("\nPout controller active (already in pose)")
    time.sleep(1)

    print("\n" + "=" * 60)
    print("ROTATION SEQUENCE")
    print("=" * 60)
    print("\nWatch Laura's body rotate while head stays in pout pose!")

    # Test rotation sequence
    angles = [0, 45, 90, -90, -45, 0]

    for i, angle in enumerate(angles, 1):
        print(f"\n[{i}/{len(angles)}] Rotating to {angle:+4d}°")

        # Update pout move target
        pout_move.rotate_to(angle)

        # Wait for rotation to complete
        print(f"   Waiting 4 seconds for rotation...")
        time.sleep(4)

        print(f"   ✓ Target set to {angle}°")

    print("\n" + "=" * 60)
    print("ROTATION TEST COMPLETE")
    print("=" * 60)

    print("\nVisual verification:")
    print("  ✓ Did Laura rotate through all angles?")
    print("  ✓ Did head stay in pout pose (not lifting up)?")
    print("  ✓ Did antennas stay spread throughout?")
    print("  ✓ Were rotations smooth?")

    print("\n✅ Test sequence finished")
    print("\nLaura should now be facing forward (0°) in pout pose.")
    print("Press Ctrl+C to stop, or wait 5 seconds...")

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
        test_pout_rotation_simple()
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
