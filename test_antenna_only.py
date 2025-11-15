#!/usr/bin/env python3
"""Test antenna twitch movement only (no pout pose, no body rotation)"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from daemon_client import DaemonClient
from moves import MovementManager, AntennaTwitchMove

def test_antenna_only():
    print("=" * 60)
    print("TEST: Antenna Twitch Only")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    time.sleep(1)

    # Test antenna twitch without any other movement
    print("\n[TEST] Starting antenna twitch pattern...")
    print("        Base position: (-3.05, +3.05) - both antennas fully down")
    print("        Flick target: (-2.5, +2.5) - both antennas partially raised")
    print("        Pattern: Random flicks every 1.0-1.5 seconds")

    antenna_twitch = AntennaTwitchMove(
        pattern="frustration_twitch",
        base_antennas=(-3.05, 3.05)  # Sleep antenna position
    )

    movement_manager.clear_move_queue()
    movement_manager.queue_move(antenna_twitch)

    print("\n   Observing antenna movement for 15 seconds...")
    for i in range(15):
        time.sleep(1)
        print(f"      {i+1}s", end="\r")

    print("\n\n✅ Test complete")
    print("=" * 60)

    movement_manager.stop()

if __name__ == "__main__":
    test_antenna_only()
