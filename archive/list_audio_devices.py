#!/usr/bin/env python3
"""
PyAudio Device Enumeration Script
Lists all available audio input devices with their indices and properties.
"""

import pyaudio

def list_audio_devices():
    """List all available PyAudio input devices."""
    p = pyaudio.PyAudio()

    print("\n" + "="*80)
    print("PyAudio Device List")
    print("="*80 + "\n")

    default_input = p.get_default_input_device_info()
    print(f"DEFAULT INPUT DEVICE: {default_input['name']} (index {default_input['index']})")
    print()

    device_count = p.get_device_count()
    print(f"Total devices found: {device_count}\n")

    for i in range(device_count):
        try:
            device_info = p.get_device_info_by_index(i)

            # Only show devices with input channels
            if device_info['maxInputChannels'] > 0:
                print(f"Device {i}: {device_info['name']}")
                print(f"  Max Input Channels: {device_info['maxInputChannels']}")
                print(f"  Default Sample Rate: {device_info['defaultSampleRate']} Hz")
                print(f"  Host API: {p.get_host_api_info_by_index(device_info['hostApi'])['name']}")

                # Highlight if this is the default input
                if i == default_input['index']:
                    print(f"  *** THIS IS THE DEFAULT INPUT DEVICE ***")

                # Highlight if this looks like the Reachy microphone
                if 'reachy' in device_info['name'].lower() or 'mini audio' in device_info['name'].lower():
                    print(f"  >>> LIKELY CORRECT DEVICE FOR REACHY <<<")

                print()
        except Exception as e:
            print(f"Device {i}: Error getting info - {e}\n")

    p.terminate()
    print("="*80)

if __name__ == "__main__":
    list_audio_devices()
