#!/usr/bin/env python3
"""Debug antenna twitch - add logging to see what's happening"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Enable debug logging
logging.basicConfig(level=logging.INFO)

from daemon_client import DaemonClient
from moves import MovementManager, AntennaTwitchMove

def test_antenna_debug():
    print("=" * 60)
    print("DEBUG: Antenna Twitch with Logging")
    print("=" * 60)

    daemon = DaemonClient("http://localhost:8100")
    movement_manager = MovementManager(current_robot=daemon, camera_worker=None)
    movement_manager.start()

    time.sleep(1)

    print("\n[TEST] Creating antenna twitch move...")
    antenna_twitch = AntennaTwitchMove(
        pattern="frustration_twitch",
        base_antennas=(-3.05, 3.05)
    )

    # Manually test evaluate to see what it returns
    print("\n[DEBUG] Manual evaluate test:")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        head, antennas, body_yaw = antenna_twitch.evaluate(t)
        print(f"  t={t:.1f}s: head={head}, antennas={antennas}, body_yaw={body_yaw}")

    print("\n[TEST] Queueing antenna twitch in MovementManager...")
    movement_manager.clear_move_queue()
    movement_manager.queue_move(antenna_twitch)

    print("   Observing for 10 seconds...")
    time.sleep(10)

    print("\n✅ Test complete")
    print("=" * 60)

    movement_manager.stop()

if __name__ == "__main__":
    test_antenna_debug()
