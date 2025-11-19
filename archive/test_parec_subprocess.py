#!/usr/bin/env python3
"""Minimal test of parec subprocess to isolate the issue."""

import asyncio
import sys

async def test_parec():
    monitor_source = "alsa_output.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_202000386253800122-00.analog-stereo.monitor"

    parec_command = [
        "parec",
        f"--device={monitor_source}",
        "--format=s16le",
        "--rate=22050",
        "--channels=2",
    ]

    print(f"Starting parec subprocess...")
    print(f"Command: {' '.join(parec_command)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *parec_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        print(f"Process started (PID: {process.pid})")

        # Try to read a small chunk
        chunk_size = 4096  # Small test chunk

        print(f"Attempting to read {chunk_size} bytes...")

        try:
            audio_bytes = await asyncio.wait_for(
                process.stdout.readexactly(chunk_size),
                timeout=5.0
            )
            print(f"✓ Successfully read {len(audio_bytes)} bytes")

        except asyncio.TimeoutError:
            print("✗ Timeout waiting for audio data")
            # Check stderr
            try:
                stderr_bytes = await asyncio.wait_for(process.stderr.read(), timeout=0.5)
                if stderr_bytes:
                    print(f"stderr: {stderr_bytes.decode()}")
            except:
                pass

        except asyncio.IncompleteReadError as e:
            print(f"✗ Stream ended prematurely (got {len(e.partial)} bytes)")
            # Check stderr
            try:
                stderr_bytes = await asyncio.wait_for(process.stderr.read(), timeout=0.5)
                if stderr_bytes:
                    print(f"stderr: {stderr_bytes.decode()}")
            except:
                pass

        # Cleanup
        print("Terminating process...")
        process.terminate()
        await process.wait()
        print("Process terminated")

    except FileNotFoundError:
        print("✗ parec command not found")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(test_parec()))
