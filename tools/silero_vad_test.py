#!/usr/bin/env python3
"""
Silero VAD Calibration Tool

Interactive tool to find the optimal Silero VAD threshold for your environment.
Tests background noise levels, speech patterns, and recommends threshold settings.
"""

import sys
import os
import time
import numpy as np
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_capture.silero_vad_wrapper import SileroVAD
from system.audio_manager import AudioManager
from tools.vad_settings import load_vad_settings, save_vad_settings

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback: no colors
    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = WHITE = RESET = ""
    class Style:
        BRIGHT = RESET_ALL = ""


class SileroVADTester:
    """Interactive Silero VAD calibration and testing tool"""

    def __init__(self):
        print("Initializing Silero VAD Tester...")

        # Initialize Silero VAD
        self.silero_vad = SileroVAD(sample_rate=16000, threshold=0.5)

        # Initialize audio manager
        self.audio_manager = AudioManager()

        # Load current settings
        try:
            self.current_settings = load_vad_settings()
            self.current_threshold = self.current_settings.get('silero_threshold', 0.5)
        except:
            self.current_threshold = 0.5
            self.current_settings = {'silero_threshold': 0.5}

        print(f"{Fore.GREEN}Ready!{Style.RESET_ALL}\n")

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')

    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "═" * 60)
        print(f"  {Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
        print("═" * 60 + "\n")

    def print_menu(self):
        """Print main menu"""
        self.clear_screen()
        print(f"\n{Fore.CYAN}{Style.BRIGHT}╔════════════════════════════════════════════════════════════╗")
        print(f"║           SILERO VAD CALIBRATION TOOL                     ║")
        print(f"╚════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

        print(f"Current Settings:")
        print(f"  Threshold: {Fore.YELLOW}{self.current_threshold:.2f}{Style.RESET_ALL}\n")

        print(f"Modes:")
        print(f"  {Fore.GREEN}1.{Style.RESET_ALL} Real-time Monitor")
        print(f"  {Fore.GREEN}2.{Style.RESET_ALL} Silence Calibration (10s)")
        print(f"  {Fore.GREEN}3.{Style.RESET_ALL} Speech Calibration (10s)")
        print(f"  {Fore.GREEN}4.{Style.RESET_ALL} Threshold Test (40s total)")
        print(f"  {Fore.GREEN}5.{Style.RESET_ALL} Update Settings")
        print(f"  {Fore.GREEN}6.{Style.RESET_ALL} Exit\n")

    def process_frame(self, pcm_bytes):
        """Process a 2048-sample frame through Silero VAD (matches speech_processor.py)"""
        frame_data_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Split 2048-sample frame into 4 chunks of 512 samples each
        chunks = [frame_data_int16[i:i+512] for i in range(0, 2048, 512)]
        speech_probabilities = [self.silero_vad.process_chunk(chunk) for chunk in chunks]

        max_probability = max(speech_probabilities)

        return speech_probabilities, max_probability

    def colorize_probability(self, prob):
        """Return colored probability string"""
        if prob < 0.3:
            color = Fore.GREEN
        elif prob < 0.7:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        return f"{color}{prob:.3f}{Style.RESET_ALL}"

    async def real_time_monitor(self):
        """Mode 1: Real-time probability monitor"""
        self.print_header("REAL-TIME MONITOR")

        print("Showing live Silero probabilities...")
        print(f"{Fore.GREEN}GREEN{Style.RESET_ALL} = silence (< 0.3)")
        print(f"{Fore.YELLOW}YELLOW{Style.RESET_ALL} = borderline (0.3-0.7)")
        print(f"{Fore.RED}RED{Style.RESET_ALL} = speech (> 0.7)")
        print(f"\nPress {Fore.CYAN}Ctrl+C{Style.RESET_ALL} to stop\n")

        # Start audio
        await self.audio_manager.start_listening()

        all_probs = []
        start_time = time.time()

        try:
            while True:
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.005)
                    continue

                chunks_probs, max_prob = self.process_frame(pcm_bytes)
                all_probs.append(max_prob)

                elapsed = time.time() - start_time

                # Format chunk probabilities
                chunks_str = "[" + ", ".join([self.colorize_probability(p) for p in chunks_probs]) + "]"

                # Determine status
                if max_prob < 0.3:
                    status = f"{Fore.GREEN}SILENCE{Style.RESET_ALL}"
                elif max_prob < 0.7:
                    status = f"{Fore.YELLOW}BORDERLINE{Style.RESET_ALL}"
                else:
                    status = f"{Fore.RED}SPEECH{Style.RESET_ALL}"

                print(f"Time: {elapsed:5.1f}s | Chunks: {chunks_str} MAX: {self.colorize_probability(max_prob)} | {status}")

                # Show stats every 10 seconds
                if len(all_probs) % 78 == 0 and len(all_probs) > 0:  # ~10 seconds at 7.8 fps
                    print(f"\n{Fore.CYAN}Statistics (last 10s):{Style.RESET_ALL}")
                    print(f"  Min: {min(all_probs[-78:]):.3f}  Max: {max(all_probs[-78:]):.3f}  " +
                          f"Mean: {np.mean(all_probs[-78:]):.3f}  Median: {np.median(all_probs[-78:]):.3f}\n")

        except KeyboardInterrupt:
            print(f"\n{Fore.GREEN}Stopped.{Style.RESET_ALL}\n")
        finally:
            await self.audio_manager.stop_listening()

            if all_probs:
                print(f"\n{Fore.CYAN}Overall Statistics:{Style.RESET_ALL}")
                print(f"  Frames: {len(all_probs)}")
                print(f"  Min: {min(all_probs):.3f}")
                print(f"  Max: {max(all_probs):.3f}")
                print(f"  Mean: {np.mean(all_probs):.3f}")
                print(f"  Median: {np.median(all_probs):.3f}")

        input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to return to menu...")

    async def silence_calibration(self):
        """Mode 2: Measure background noise levels"""
        self.print_header("SILENCE CALIBRATION")

        duration = 10.0
        print(f"Recording {duration:.0f} seconds of ambient silence...")
        print(f"{Fore.YELLOW}Please remain quiet.{Style.RESET_ALL}\n")

        input(f"Press {Fore.CYAN}Enter{Style.RESET_ALL} when ready to start...")

        # Start audio
        await self.audio_manager.start_listening()

        probabilities = []
        start_time = time.time()

        print("\nRecording", end="", flush=True)

        try:
            while time.time() - start_time < duration:
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.005)
                    continue

                _, max_prob = self.process_frame(pcm_bytes)
                probabilities.append(max_prob)

                # Progress dots
                if len(probabilities) % 8 == 0:
                    print(".", end="", flush=True)

        finally:
            await self.audio_manager.stop_listening()
            print(f" {Fore.GREEN}Done!{Style.RESET_ALL}\n")

        # Analyze results
        self.print_silence_results(probabilities)

        input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to return to menu...")

    def print_silence_results(self, probabilities):
        """Print analysis of silence calibration"""
        if not probabilities:
            print(f"{Fore.RED}No data collected!{Style.RESET_ALL}")
            return

        min_prob = min(probabilities)
        max_prob = max(probabilities)
        mean_prob = np.mean(probabilities)
        median_prob = np.median(probabilities)
        p95 = np.percentile(probabilities, 95)

        print(f"{Fore.CYAN}Results:{Style.RESET_ALL}")
        print(f"  Frames analyzed: {len(probabilities)}")
        print(f"  Min: {min_prob:.3f}")
        print(f"  Max: {max_prob:.3f}  {Fore.RED if max_prob > 0.7 else Fore.YELLOW if max_prob > 0.5 else ''}{'⚠️  Outlier!' if max_prob > 0.7 else ''}{Style.RESET_ALL}")
        print(f"  Mean: {mean_prob:.3f}")
        print(f"  Median: {median_prob:.3f}")
        print(f"  95th percentile: {p95:.3f}")

        # Histogram
        print(f"\n{Fore.CYAN}Distribution:{Style.RESET_ALL}")
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(probabilities, bins=bins)

        for i in range(len(hist)):
            bin_start = bins[i]
            bin_end = bins[i+1]
            count = hist[i]
            pct = (count / len(probabilities)) * 100

            bar_length = int(pct / 2)  # Scale to fit terminal
            bar = "█" * bar_length

            print(f"  {bin_start:.1f}-{bin_end:.1f}: {bar} ({count} frames, {pct:.1f}%)")

        # Recommendation
        print(f"\n{Fore.CYAN}RECOMMENDED THRESHOLD:{Style.RESET_ALL} {Fore.YELLOW}{max(max_prob + 0.1, p95 + 0.05):.2f}{Style.RESET_ALL}")
        print(f"  Reasoning: Max silence ({max_prob:.3f}) + safety margin")

        if max_prob > 0.7:
            print(f"\n{Fore.RED}⚠️  WARNING:{Style.RESET_ALL} Very high noise detected!")
            print(f"  Your environment may have constant background noise classified as speech.")
            print(f"  Consider testing with consecutive frame requirements to reduce false positives.")

    async def speech_calibration(self):
        """Mode 3: Measure speech probability levels"""
        self.print_header("SPEECH CALIBRATION")

        duration = 10.0
        print(f"Recording {duration:.0f} seconds of normal speech...")
        print(f"{Fore.YELLOW}Please speak continuously at normal volume.{Style.RESET_ALL}\n")

        input(f"Press {Fore.CYAN}Enter{Style.RESET_ALL} when ready to start...")

        # Start audio
        await self.audio_manager.start_listening()

        probabilities = []
        start_time = time.time()

        print("\nRecording", end="", flush=True)

        try:
            while time.time() - start_time < duration:
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.005)
                    continue

                _, max_prob = self.process_frame(pcm_bytes)
                probabilities.append(max_prob)

                # Progress dots
                if len(probabilities) % 8 == 0:
                    print(".", end="", flush=True)

        finally:
            await self.audio_manager.stop_listening()
            print(f" {Fore.GREEN}Done!{Style.RESET_ALL}\n")

        # Analyze results
        self.print_speech_results(probabilities)

        input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to return to menu...")

    def print_speech_results(self, probabilities):
        """Print analysis of speech calibration"""
        if not probabilities:
            print(f"{Fore.RED}No data collected!{Style.RESET_ALL}")
            return

        min_prob = min(probabilities)
        max_prob = max(probabilities)
        mean_prob = np.mean(probabilities)
        median_prob = np.median(probabilities)
        p05 = np.percentile(probabilities, 5)

        print(f"{Fore.CYAN}Results:{Style.RESET_ALL}")
        print(f"  Frames analyzed: {len(probabilities)}")
        print(f"  Min: {min_prob:.3f}  {Fore.YELLOW if min_prob < 0.3 else ''}{'⚠️  Very low!' if min_prob < 0.3 else ''}{Style.RESET_ALL}")
        print(f"  Max: {max_prob:.3f}")
        print(f"  Mean: {mean_prob:.3f}")
        print(f"  Median: {median_prob:.3f}")
        print(f"  5th percentile: {p05:.3f}")

        # Histogram
        print(f"\n{Fore.CYAN}Distribution:{Style.RESET_ALL}")
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(probabilities, bins=bins)

        for i in range(len(hist)):
            bin_start = bins[i]
            bin_end = bins[i+1]
            count = hist[i]
            pct = (count / len(probabilities)) * 100

            bar_length = int(pct / 2)
            bar = "█" * bar_length

            print(f"  {bin_start:.1f}-{bin_end:.1f}: {bar} ({count} frames, {pct:.1f}%)")

        if min_prob < 0.3:
            print(f"\n{Fore.YELLOW}Note:{Style.RESET_ALL} Some speech frames had low probability.")
            print(f"  This is normal for pauses, breath sounds, or soft phonemes.")

    async def threshold_test(self):
        """Mode 4: Test multiple thresholds with silence and speech"""
        self.print_header("THRESHOLD TEST")

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        silence_duration = 10.0
        speech_duration = 10.0

        print(f"Testing thresholds: {', '.join([str(t) for t in thresholds])}")
        print(f"\nPhase 1: Silence ({silence_duration:.0f}s)")
        print(f"Phase 2: Speech ({speech_duration:.0f}s)\n")

        # Phase 1: Silence
        print(f"{Fore.YELLOW}Phase 1: Please remain silent{Style.RESET_ALL}")
        input(f"Press {Fore.CYAN}Enter{Style.RESET_ALL} when ready...")

        await self.audio_manager.start_listening()
        silence_probs = []
        start_time = time.time()

        print("Recording silence", end="", flush=True)

        try:
            while time.time() - start_time < silence_duration:
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.005)
                    continue

                _, max_prob = self.process_frame(pcm_bytes)
                silence_probs.append(max_prob)

                if len(silence_probs) % 8 == 0:
                    print(".", end="", flush=True)
        finally:
            await self.audio_manager.stop_listening()
            print(f" {Fore.GREEN}Done!{Style.RESET_ALL}\n")

        # Phase 2: Speech
        print(f"{Fore.YELLOW}Phase 2: Please speak continuously{Style.RESET_ALL}")
        input(f"Press {Fore.CYAN}Enter{Style.RESET_ALL} when ready...")

        await self.audio_manager.start_listening()
        speech_probs = []
        start_time = time.time()

        print("Recording speech", end="", flush=True)

        try:
            while time.time() - start_time < speech_duration:
                pcm_bytes = self.audio_manager.read_audio_frame()
                if not pcm_bytes:
                    await asyncio.sleep(0.005)
                    continue

                _, max_prob = self.process_frame(pcm_bytes)
                speech_probs.append(max_prob)

                if len(speech_probs) % 8 == 0:
                    print(".", end="", flush=True)
        finally:
            await self.audio_manager.stop_listening()
            print(f" {Fore.GREEN}Done!{Style.RESET_ALL}\n")

        # Analyze results
        self.print_threshold_test_results(thresholds, silence_probs, speech_probs)

        input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to return to menu...")

    def print_threshold_test_results(self, thresholds, silence_probs, speech_probs):
        """Print threshold test analysis"""
        print(f"{Fore.CYAN}Silence Phase Results:{Style.RESET_ALL}")

        silence_results = {}
        for threshold in thresholds:
            false_positives = sum(1 for p in silence_probs if p > threshold)
            fp_rate = (false_positives / len(silence_probs)) * 100
            silence_results[threshold] = (false_positives, fp_rate)

            status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if fp_rate < 5 else f"{Fore.RED}✗{Style.RESET_ALL}"
            print(f"  {threshold:.1f}: {status} {false_positives} false positives ({fp_rate:.1f}%)")

        print(f"\n{Fore.CYAN}Speech Phase Results:{Style.RESET_ALL}")

        speech_results = {}
        for threshold in thresholds:
            detections = sum(1 for p in speech_probs if p > threshold)
            detection_rate = (detections / len(speech_probs)) * 100
            speech_results[threshold] = (detections, detection_rate)

            status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if detection_rate > 80 else f"{Fore.RED}✗{Style.RESET_ALL}"
            print(f"  {threshold:.1f}: {status} Detected ({detection_rate:.0f}% frames)")

        # Find best threshold
        print(f"\n{Fore.CYAN}RECOMMENDATION:{Style.RESET_ALL}")

        best_threshold = None
        best_score = -1

        for threshold in thresholds:
            fp_rate = silence_results[threshold][1]
            detection_rate = speech_results[threshold][1]

            # Score: prioritize low false positives, then high detection
            score = detection_rate - (fp_rate * 5)  # Penalize FPs heavily

            if score > best_score:
                best_score = score
                best_threshold = threshold

        fp_rate = silence_results[best_threshold][1]
        detection_rate = speech_results[best_threshold][1]

        print(f"  Threshold: {Fore.YELLOW}{best_threshold:.1f}{Style.RESET_ALL}")
        print(f"  False positive rate: {fp_rate:.1f}%")
        print(f"  Speech detection rate: {detection_rate:.0f}%")
        print(f"  Reasoning: Best balance for your environment")

        self.recommended_threshold = best_threshold

    def update_settings(self):
        """Mode 5: Update VAD settings file"""
        self.print_header("UPDATE SETTINGS")

        print(f"Current threshold: {Fore.YELLOW}{self.current_threshold:.2f}{Style.RESET_ALL}")

        if hasattr(self, 'recommended_threshold'):
            print(f"Recommended threshold: {Fore.GREEN}{self.recommended_threshold:.2f}{Style.RESET_ALL}")
            new_threshold = self.recommended_threshold
        else:
            print(f"\n{Fore.YELLOW}No recommendation available.{Style.RESET_ALL}")
            print("Run Threshold Test (option 4) first to get a recommendation.\n")

            try:
                new_threshold = float(input("Enter new threshold (0.0-1.0) or press Enter to cancel: "))
                if new_threshold < 0.0 or new_threshold > 1.0:
                    print(f"{Fore.RED}Invalid threshold. Must be between 0.0 and 1.0.{Style.RESET_ALL}")
                    input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to return to menu...")
                    return
            except ValueError:
                print("Cancelled.")
                input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to return to menu...")
                return

        # Confirm
        print(f"\nApply threshold {Fore.YELLOW}{new_threshold:.2f}{Style.RESET_ALL}?")
        confirm = input("[y/N]: ").strip().lower()

        if confirm == 'y':
            # Update settings
            self.current_settings['silero_threshold'] = new_threshold
            save_vad_settings(self.current_settings, 'current')

            self.current_threshold = new_threshold

            print(f"{Fore.GREEN}Settings updated successfully!{Style.RESET_ALL}")
            print(f"Threshold set to {new_threshold:.2f}")
        else:
            print("Cancelled.")

        input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to return to menu...")

    async def run(self):
        """Main menu loop"""
        while True:
            self.print_menu()

            try:
                choice = input(f"Select [1-6]: ").strip()

                if choice == '1':
                    await self.real_time_monitor()
                elif choice == '2':
                    await self.silence_calibration()
                elif choice == '3':
                    await self.speech_calibration()
                elif choice == '4':
                    await self.threshold_test()
                elif choice == '5':
                    self.update_settings()
                elif choice == '6':
                    print(f"\n{Fore.GREEN}Goodbye!{Style.RESET_ALL}\n")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please select 1-6.{Style.RESET_ALL}")
                    time.sleep(1)

            except KeyboardInterrupt:
                print(f"\n\n{Fore.GREEN}Goodbye!{Style.RESET_ALL}\n")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()
                input(f"\nPress {Fore.CYAN}Enter{Style.RESET_ALL} to continue...")


async def main():
    """Entry point"""
    tester = SileroVADTester()
    await tester.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
