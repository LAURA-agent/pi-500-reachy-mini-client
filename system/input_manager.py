#!/usr/bin/env python3

import asyncio
import time
import random
import select
from datetime import datetime
from pathlib import Path
from typing import Optional
from evdev import InputDevice, list_devices, ecodes
from colorama import Fore

# Wake word sensitivity settings (0.0-1.0, higher = more sensitive but more false positives)
WAKE_WORDS = {
    "Laura.pmdl": 0.6,           # Main wake word - slightly more sensitive
    "Wake_up_Laura.pmdl": 0.5,    # Wake from sleep
    "GD_Laura.pmdl": 0.5,         # Pout mode trigger
    "tookmycrazypills.pmdl": 0.5, # Medicine acknowledgment
    "claudecode.pmdl": 0.5,       # Claude Code wake
    "sendnote.pmdl": 0.5,         # Send note command
}

class InputManager:
    """
    Manages all input detection including keyboard and wake word monitoring.
    
    Handles wake event detection, keyboard monitoring, and coordination
    between different input sources.
    """
    
    def __init__(self, audio_manager):
        self.audio_manager = audio_manager
        self.keyboard_device = None
        self.last_interaction = datetime.now()
        self.last_interaction_check = datetime.now()
        
        # Key state tracking for modifier detection
        self.keys_pressed = set()
        
        # Wake word detection attributes
        self.wake_detector = None
        self.wake_model_names = None
        self.wake_pa = None
        self.wake_stream = None
        self.wake_last_break = None
        
    def find_pi_keyboard(self):
        """Find the Pi 500 keyboard device with proper priority and logging"""
        if not InputDevice:
            print(f"{Fore.YELLOW}[WARN] evdev not available, keyboard detection disabled{Fore.WHITE}")
            return None

        keyboard_devices = []
        for path in list_devices():
            try:
                device = InputDevice(path)

                # Check if we can read from this device
                try:
                    select.select([device.fd], [], [], 0)

                    # Check for Pi 500 keyboard specifically
                    if "Pi 500" in device.name and "Keyboard" in device.name:
                        # Skip Mouse, Consumer Control, and System Control devices
                        if "Mouse" in device.name or "Consumer" in device.name or "System" in device.name:
                            device.close()
                            continue

                        # Assign priority to actual keyboard devices
                        # IMPORTANT: On Pi 500, event9 is the ONLY device that receives KEY_LEFTMETA
                        # - event9: Receives left meta (Raspberry button) key events
                        # - event13: Receives other keyboard keys but NOT left meta
                        # - event10: Mouse device (filtered out above)
                        # - event11: System Control (filtered out above)
                        # - event12: Consumer Control (filtered out above)
                        priority = 0
                        if "event9" in device.path:
                            priority = 100  # Top priority - this device receives KEY_LEFTMETA
                        elif "event5" in device.path:
                            priority = 90   # Fallback (doesn't exist on current Pi 500)
                        elif "event13" in device.path:
                            priority = 80   # Secondary keyboard device (no left meta events)
                        else:
                            priority = 50   # Other keyboard devices

                        if priority > 0:
                            keyboard_devices.append((device, priority))
                    else:
                        device.close()

                except Exception:
                    device.close()

            except Exception:
                pass

        if keyboard_devices:
            keyboard_devices.sort(key=lambda x: x[1], reverse=True)
            keyboard_device = keyboard_devices[0][0]
            print(f"{Fore.GREEN}✓ Keyboard: {keyboard_device.path} ({keyboard_device.name}){Fore.WHITE}")
            return keyboard_device
        else:
            print(f"{Fore.YELLOW}✗ No Pi 500 Keyboard found{Fore.WHITE}")
            return None

    def initialize_keyboard(self):
        """Initialize keyboard input detection"""
        self.keyboard_device = self.find_pi_keyboard()
        return self.keyboard_device is not None

    def _listen_keyboard_sync(self) -> str | None:
        """Synchronous keyboard check for wake event"""
        if not self.keyboard_device or not ecodes:
            return None

        try:
            ready, _, _ = select.select([self.keyboard_device.fd], [], [], 0.001)
            if ready:
                for event in self.keyboard_device.read():
                    if event.type == ecodes.EV_KEY:
                        # Track key state changes
                        if event.value == 1:  # Key press
                            self.keys_pressed.add(event.code)
                        elif event.value == 0:  # Key release
                            self.keys_pressed.discard(event.code)
                        
                        # Check for left meta press
                        if (event.code == ecodes.KEY_LEFTMETA and event.value == 1):
                            # Check if shift is currently held
                            if ecodes.KEY_LEFTSHIFT in self.keys_pressed:
                                print("[INFO] SHIFT+Left Meta detected - routing to Claude Code")
                                return "keyboard_code"
                            else:
                                print("[INFO] Left Meta detected - routing to LAURA")
                                return "keyboard_laura"
                                
        except (BlockingIOError, OSError):
            pass
            
        return None

    async def wake_word_detection(self):
        """Wake word detection with notification-aware breaks and cancellation support"""
        import pyaudio
        import snowboydetect
        from config.client_config import WAKE_WORDS_AND_SENSITIVITIES as WAKE_WORDS, WAKEWORD_RESOURCE_FILE, BASE_PATH

        # One-time initialization
        if not self.wake_detector:
            try:
                print(f"{Fore.YELLOW}Initializing wake word detector...{Fore.WHITE}")

                # Explicitly define resource path
                resource_path = Path(WAKEWORD_RESOURCE_FILE)

                # Set the directory where all wake word models are kept
                wakeword_dir = BASE_PATH / "wakewords"

                # Build model paths from filenames in WAKE_WORDS
                model_paths = [wakeword_dir / name for name in WAKE_WORDS.keys()]

                # Check for missing files
                missing = [str(path.absolute()) for path in [resource_path] + model_paths if not path.exists()]
                if missing:
                    print(f"ERROR: The following required file(s) are missing:\n" + "\n".join(missing))
                    return None

                # Build sensitivities list, ensuring order matches models
                sensitivities = []
                for p in model_paths:
                    sensitivity = WAKE_WORDS.get(p.name)
                    if sensitivity is None:
                        print(f"WARNING: No sensitivity found for {p.name}. Defaulting to 0.5.")
                        sensitivity = 0.5
                    sensitivities.append(str(sensitivity))
                if len(sensitivities) != len(model_paths):
                    print("ERROR: Sensitivities count does not match model paths count!")
                    return None

                # Initialize the detector
                self.wake_detector = snowboydetect.SnowboyDetect(
                    resource_filename=str(resource_path.absolute()).encode(),
                    model_str=",".join(str(p.absolute()) for p in model_paths).encode()
                )
                sensitivity_bytes = ",".join(sensitivities).encode()
                self.wake_detector.SetSensitivity(sensitivity_bytes)
                self.wake_model_names = [p.name for p in model_paths]
                self.wake_pa = pyaudio.PyAudio()
                self.wake_stream = None
                self.wake_last_break = time.time()
                print(f"{Fore.GREEN}Wake word detector initialized with models: {self.wake_model_names}{Fore.WHITE}")
            except Exception as e:
                print(f"Error initializing wake word detection: {e}")
                return None

        try:
            # Create/restart stream if needed
            if not self.wake_stream or not self.wake_stream.is_active():
                # Use reSpeaker XVF3800 4-Mic Array directly via ALSA
                # reSpeaker natively supports 16kHz, no resampling needed
                input_device_index = None
                self.wake_input_channels = 2  # reSpeaker outputs stereo (2 channels)
                for i in range(self.wake_pa.get_device_count()):
                    info = self.wake_pa.get_device_info_by_index(i)
                    # Look for reSpeaker XVF3800
                    if 'reSpeaker' in info['name'] and info['maxInputChannels'] > 0:
                        input_device_index = i
                        self.wake_input_channels = 2  # Stereo, will downmix to mono
                        print(f"[Wake Word] Using reSpeaker XVF3800 (device {i}) at native 16kHz with {self.wake_input_channels} channels")
                        break

                if input_device_index is None:
                    print("[Wake Word] reSpeaker not found, using default device")

                self.wake_stream = self.wake_pa.open(
                    rate=16000,
                    channels=self.wake_input_channels,
                    format=pyaudio.paInt16,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=1024
                )
                self.wake_stream.start_stream()

            # Periodic breaks for notifications (every 20 seconds)
            current_time = time.time()
            if (current_time - self.wake_last_break) >= 20:
                self.wake_last_break = current_time
                if self.wake_stream:
                    self.wake_stream.stop_stream()
                    await asyncio.sleep(1)  # 1-second break
                    self.wake_stream.start_stream()
                return None

            # Read audio with error handling
            # Use asyncio.to_thread to prevent blocking the event loop (allows Ctrl+C to work)
            try:
                data = await asyncio.to_thread(
                    self.wake_stream.read, 1024, exception_on_overflow=False
                )
                if len(data) == 0:
                    print("Warning: Empty audio frame received")
                    return None

                # If stereo (2 channels), downmix to mono for Snowboy
                if self.wake_input_channels == 2:
                    import numpy as np
                    # Convert bytes to int16 array
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    # Reshape to (samples, channels) and average channels
                    audio_data = audio_data.reshape(-1, 2)
                    mono_data = audio_data.mean(axis=1, dtype=np.int16)
                    # Convert back to bytes
                    data = mono_data.tobytes()

            except (IOError, OSError) as e:
                print(f"Stream read error: {e}")
                if self.wake_stream:
                    self.wake_stream.stop_stream()
                    self.wake_stream.close()
                    self.wake_stream = None
                return None
            except asyncio.CancelledError:
                # Allow clean cancellation during shutdown
                print("[INFO] Wake word detection cancelled")
                if self.wake_stream:
                    self.wake_stream.stop_stream()
                return None

            result = self.wake_detector.RunDetection(data)
            if result > 0:
                print(f"{Fore.GREEN}Wake word detected! (Model {result}){Fore.WHITE}")
                self.last_interaction = datetime.now()
                self.last_interaction_check = datetime.now()
                return self.wake_model_names[result-1] if result <= len(self.wake_model_names) else None

            # Yield to event loop more frequently to allow cancellation
            await asyncio.sleep(0)

            return None

        except Exception as e:
            print(f"Error in wake word detection: {e}")
            if self.wake_stream:
                self.wake_stream.stop_stream()
                self.wake_stream.close()
                self.wake_stream = None
            return None

    def get_time_since_last_interaction(self):
        """Get time in seconds since last interaction."""
        return (datetime.now() - self.last_interaction).total_seconds()

    def update_last_interaction(self):
        """Update the last interaction timestamp to now."""
        self.last_interaction = datetime.now()
        self.last_interaction_check = datetime.now()

    def set_keyboard_cooldown(self, duration: float):
        """Set a cooldown period after keyboard input to prevent re-triggers."""
        # Simple implementation: just update last interaction time
        # This prevents immediate re-trigger from held keys
        self.last_interaction = datetime.now()

    async def check_for_wake_events(self):
        """Check for wake events from keyboard or wake word"""
        wake_event_source = None

        # Check keyboard first
        keyboard_event = self._listen_keyboard_sync()
        if keyboard_event:
            wake_event_source = keyboard_event  # Can be "keyboard_laura" or "keyboard_code"
            print(f"[INFO] Wake event from keyboard: {keyboard_event}")
        else:
            # Check wake word
            wakeword_model = await self.wake_word_detection()
            if wakeword_model:
                wake_event_source = f"wakeword ({wakeword_model})"
                print(f"[INFO] Wake event from: {wake_event_source}")

        return wake_event_source

    def stop_wake_word_detection(self):
        """Temporarily stop wake word detection (e.g., during voice input)"""
        try:
            if self.wake_stream and self.wake_stream.is_active():
                self.wake_stream.stop_stream()
                # Close and recreate stream to avoid segfault on restart
                self.wake_stream.close()
                self.wake_stream = None
        except Exception as e:
            print(f"[Wake Word] Error stopping stream: {e}")
            self.wake_stream = None

    def resume_wake_word_detection(self):
        """Resume wake word detection after temporary pause"""
        # Stream will be recreated on next wake_word_detection() call
        # This avoids segfaults from trying to restart closed streams
        pass

    def restart_wake_word_detection(self):
        """Alias for resume_wake_word_detection for compatibility"""
        self.resume_wake_word_detection()

    def cleanup(self):
        """Clean up keyboard and wake word resources"""
        if self.keyboard_device:
            self.keyboard_device.close()
            self.keyboard_device = None

        # Clean up wake word stream
        if self.wake_stream:
            if self.wake_stream.is_active():
                self.wake_stream.stop_stream()
            self.wake_stream.close()
            self.wake_stream = None

        if self.wake_pa:
            self.wake_pa.terminate()
            self.wake_pa = None