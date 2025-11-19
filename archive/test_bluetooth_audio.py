#!/usr/bin/env python3
"""
Standalone test script for Bluetooth audio reactor.

Tests real-time audio capture from Bluetooth and motion synchronization
without running the full run_pi_reachy.py system.

Usage:
    python3 test_bluetooth_audio.py

Then play audio from your iPhone Claude app via Bluetooth.
Press Ctrl+C to stop.
"""

import asyncio
import signal
import sys
from daemon_client import DaemonClient
from moves import MovementManager
from camera_worker import CameraWorker
from mediapipe_face_detector import HeadTracker
from daemon_media_wrapper import DaemonMediaWrapper
from bluetooth_audio_reactor import BluetoothAudioReactor


class MinimalAudioManager:
    """Minimal AudioManager stub for BluetoothAudioReactor compatibility."""
    def __init__(self):
        self.is_playing = False
        self.is_speaking = False
        self.is_listening = False


class BluetoothAudioTest:
    """Minimal test harness for Bluetooth audio → motion."""

    def __init__(self):
        print("[TEST] Initializing Bluetooth audio test...")

        # Connect to daemon
        print("[TEST] Connecting to Reachy daemon...")
        self.daemon_client = DaemonClient("http://localhost:8100")
        self.media_manager = DaemonMediaWrapper("http://localhost:8100")

        # Initialize head tracker (needed for camera worker)
        print("[TEST] Initializing head tracker...")
        self.head_tracker = HeadTracker(device="cpu")

        # Initialize camera worker (without face tracking)
        print("[TEST] Initializing camera worker...")
        self.camera_worker = CameraWorker(
            media_manager=self.media_manager,
            head_tracker=self.head_tracker,
            daemon_client=self.daemon_client,
            movement_manager=None,  # Will link after MovementManager creation
            debug_window=False
        )

        # Initialize movement manager
        print("[TEST] Initializing movement manager...")
        self.movement_manager = MovementManager(
            current_robot=self.daemon_client,
            camera_worker=self.camera_worker
        )

        # Link movement manager back to camera worker
        self.camera_worker.movement_manager = self.movement_manager

        # Initialize minimal audio manager stub
        self.audio_manager = MinimalAudioManager()

        # Initialize Bluetooth reactor
        print("[TEST] Initializing Bluetooth audio reactor...")
        self.bluetooth_reactor = BluetoothAudioReactor(
            self.movement_manager,
            self.audio_manager
        )

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True

        print("[TEST] Initialization complete!")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n[TEST] Received interrupt signal, shutting down...")
        self._running = False

    async def run(self):
        """Run the test."""
        try:
            # Start threads
            print("[TEST] Starting camera worker and movement manager threads...")
            self.camera_worker.start()
            self.movement_manager.start()

            # Wait for threads to be ready
            await asyncio.sleep(1.0)

            # Disable face tracking (not needed for this test)
            self.camera_worker.set_head_tracking_enabled(False)
            print("[TEST] Face tracking disabled")

            # Pause breathing (so we can see Bluetooth motion clearly)
            self.movement_manager.pause_breathing()
            print("[TEST] Breathing paused")

            # Start Bluetooth reactor
            print("\n" + "="*60)
            print("BLUETOOTH AUDIO REACTOR TEST")
            print("="*60)
            print("Starting Bluetooth audio monitoring...")
            print("Play audio from your iPhone Claude app now.")
            print("Reachy's antennas and head should move with the audio.")
            print("Press Ctrl+C to stop.")
            print("="*60 + "\n")

            await self.bluetooth_reactor.start()

            # Keep running until interrupted
            while self._running:
                # Print reactor state every 5 seconds
                await asyncio.sleep(5.0)
                state = self.bluetooth_reactor.get_state()
                if state['is_receiving_audio']:
                    print(f"[STATUS] Receiving audio - Intensity: {state['current_intensity']:.2f}")
                else:
                    print(f"[STATUS] Waiting for audio (device: {state['bluetooth_device_name']})")

        except KeyboardInterrupt:
            print("\n[TEST] Keyboard interrupt received")
        except Exception as e:
            print(f"[TEST ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        print("\n[TEST] Cleaning up...")

        # Stop Bluetooth reactor
        try:
            await self.bluetooth_reactor.stop()
            print("[TEST] Bluetooth reactor stopped")
        except Exception as e:
            print(f"[TEST] Error stopping Bluetooth reactor: {e}")

        # Resume breathing
        try:
            self.movement_manager.resume_breathing()
            print("[TEST] Breathing resumed")
        except Exception as e:
            print(f"[TEST] Error resuming breathing: {e}")

        # Stop threads
        try:
            self.movement_manager.stop()
            self.camera_worker.stop()
            print("[TEST] Threads stopped")
        except Exception as e:
            print(f"[TEST] Error stopping threads: {e}")

        print("[TEST] Cleanup complete")


async def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("REACHY BLUETOOTH AUDIO TEST")
    print("="*60 + "\n")

    # Check if daemon is running
    print("[TEST] Checking Reachy daemon connection...")
    daemon = DaemonClient("http://localhost:8100")
    try:
        # Simple ping to check connection
        status = daemon.get_status()
        print(f"[TEST] ✓ Daemon connected (status: {status.get('status', 'unknown')})\n")
    except Exception as e:
        print(f"[TEST ERROR] Cannot connect to Reachy daemon: {e}")
        print("[TEST] Make sure the daemon is running:")
        print("[TEST]   /home/user/reachy/venv/bin/reachy-mini-daemon")
        sys.exit(1)

    # Run test
    test = BluetoothAudioTest()
    await test.run()

    print("\n[TEST] Test complete. Goodbye!\n")


if __name__ == "__main__":
    asyncio.run(main())
