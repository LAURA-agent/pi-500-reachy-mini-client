#!/usr/bin/env python3
"""Monitor Bluetooth A2DP connection status and audio sink availability.

This script watches for:
1. Bluetooth card appearing in PipeWire
2. A2DP sink profile activation
3. Bluetooth audio sink becoming available
4. Audio stream starting

Run this while starting audio playback on iPhone to see when connection establishes.
"""

import subprocess
import time
import sys

def check_bluetooth_card():
    """Check if Bluetooth card is visible to PipeWire."""
    result = subprocess.run(
        ["pactl", "list", "cards", "short"],
        capture_output=True,
        text=True
    )
    for line in result.stdout.splitlines():
        if "bluez_card" in line:
            card_id, card_name, _ = line.split(None, 2)
            return card_name
    return None

def check_card_profile(card_name):
    """Check active profile for Bluetooth card."""
    result = subprocess.run(
        ["pactl", "list", "cards"],
        capture_output=True,
        text=True
    )

    in_card_section = False
    for line in result.stdout.splitlines():
        if f"Name: {card_name}" in line:
            in_card_section = True
        elif in_card_section and "Active Profile:" in line:
            return line.split(":", 1)[1].strip()
        elif in_card_section and line.startswith("Name:"):
            # Moved to next card
            break
    return None

def check_bluetooth_sinks():
    """Check for Bluetooth audio sinks."""
    result = subprocess.run(
        ["pactl", "list", "sinks", "short"],
        capture_output=True,
        text=True
    )

    bluetooth_sinks = []
    for line in result.stdout.splitlines():
        if "bluez_output" in line:
            sink_id, sink_name, _, _, state = line.split(None, 4)
            bluetooth_sinks.append((sink_name, state))
    return bluetooth_sinks

def check_audio_streams():
    """Check for active audio streams to Bluetooth."""
    result = subprocess.run(
        ["wpctl", "status"],
        capture_output=True,
        text=True
    )

    in_streams = False
    bluetooth_streams = []
    for line in result.stdout.splitlines():
        if "Streams:" in line:
            in_streams = True
        elif in_streams and "bluez" in line.lower():
            bluetooth_streams.append(line.strip())
        elif in_streams and line and not line.startswith(" "):
            # End of streams section
            break

    return bluetooth_streams

def main():
    print("Bluetooth A2DP Connection Monitor")
    print("=" * 60)
    print("\nWaiting for iPhone to start playing audio...")
    print("On your iPhone:")
    print("  1. Start playing audio (Spotify, Claude app, etc.)")
    print("  2. Select 'iPhone' or 'Raspberry Pi' as audio output")
    print("\nPress Ctrl+C to stop monitoring\n")

    last_state = None

    try:
        while True:
            # Check connection state
            card_name = check_bluetooth_card()
            profile = check_card_profile(card_name) if card_name else None
            sinks = check_bluetooth_sinks()
            streams = check_audio_streams()

            # Build current state
            current_state = {
                'card': card_name is not None,
                'profile': profile,
                'sinks': len(sinks),
                'streams': len(streams)
            }

            # Print updates only when state changes
            if current_state != last_state:
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n[{timestamp}] State Change:")
                print(f"  Bluetooth Card:   {'✓' if card_name else '✗'} {card_name or 'Not detected'}")
                print(f"  Active Profile:   {profile or 'None'}")
                print(f"  Bluetooth Sinks:  {len(sinks)}")
                for sink_name, state in sinks:
                    print(f"    - {sink_name} [{state}]")
                print(f"  Active Streams:   {len(streams)}")
                for stream in streams:
                    print(f"    - {stream}")

                # Check if A2DP is active
                if profile and "a2dp" in profile.lower() and sinks:
                    print(f"\n{'='*60}")
                    print("✓ A2DP CONNECTION ACTIVE!")
                    print(f"{'='*60}")
                    print("\nYou can now test the loopback approach:")
                    print(f"  Bluetooth sink: {sinks[0][0]}")
                    print("\nNext steps:")
                    print("  1. Run the pw-loopback command to intercept audio")
                    print("  2. Audio should play through Reachy speakers with 2s delay")
                    print("  3. Verify we can capture PCM data from virtual source")

                last_state = current_state.copy()

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
