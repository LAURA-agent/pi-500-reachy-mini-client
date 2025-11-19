# processing/command_handler.py
"""
Detects and handles special wake words and text-based system commands
that trigger specific actions instead of a standard AI conversation.
"""
import os
import subprocess
import asyncio
from utils.helpers import SOUND_BASE_PATH

class CommandHandler:
    def __init__(self, robot_controller, mcp_session_manager, system_command_manager):
        self.robot_controller = robot_controller
        self.mcp_session_manager = mcp_session_manager
        self.system_command_manager = system_command_manager
        self.session_id = None
        self.claude_code_mode_requested = False

    def set_session_id(self, session_id: str):
        self.session_id = session_id

    def _extract_model_name(self, wake_event_source: str) -> str | None:
        if '(' in wake_event_source and ')' in wake_event_source:
            return wake_event_source.split('(')[1].rstrip(')')
        return None

    async def handle_wake_command(self, wake_event_source: str) -> bool:
        """
        Handles commands triggered by specific wake words.
        Returns True if the event was a command and was handled immediately.
        Returns False if interaction should proceed to speech capture (potentially with flags set).
        """
        model_name = self._extract_model_name(wake_event_source)
        if not model_name:
            return False

        # 1. Sleep Command
        if model_name == "goodnightreachy.pmdl":
            print("[COMMAND] 'goodnight reachy' detected - entering sleep mode")
            self.robot_controller.enter_sleep_mode()
            return True # Handled immediately

        # 1.5 Sleep Interruption Wake Words (hey laura, come on laura)
        if model_name in ["heylaura.pmdl", "comeonlaura.pmdl"]:
            current_state = self.robot_controller.state_tracker.get_state()

            if current_state == "sleep":
                print(f"[COMMAND] '{model_name}' in SLEEP - playing sleepy response")
                # Disable wake words during response
                self.robot_controller.input_manager.stop_wake_word_detection()

                # Play pre-recorded sleepy response with motion (stays in sleep pose)
                await self.robot_controller.play_prerecorded_response(
                    self.robot_controller.sleep_response_cache,
                    response_type="SLEEP"
                )

                # Re-enable wake words
                self.robot_controller.input_manager.restart_wake_word_detection()

                return True # Handled immediately, stay in sleep

            # Not in sleep mode - treat as normal wake word
            print(f"[COMMAND] '{model_name}' outside sleep - normal wake")
            return False # Proceed to speech capture

        # 2. Pout/Frustration Command
        if model_name == "GD_Laura.pmdl":
            print("[COMMAND] Frustration wake word detected - entering pout mode")
            # Play random frustrated audio (logic simplified from monolith for clarity, but functionality remains)
            import glob
            import random
            frustrated_dir = "/home/user/reachy/pi_reachy_deployment/assets/sounds/laura/wake_sentences/frustrated"
            files = glob.glob(f"{frustrated_dir}/*.mp3")
            if files:
                # CRITICAL: Wait for audio to finish before returning (prevents feedback loop)
                start_time, completion_task = await self.robot_controller.audio_manager.play_audio(random.choice(files))
                await completion_task  # Block until audio finishes playing

            self.robot_controller.enter_pout_mode(entry_speed="slow")
            return True # Handled immediately

        # 3. Pout Exit / Apology (ONLY works when in pout mode)
        if model_name in ["sorrylaura.pmdl", "sorryreachy.pmdl", "quitbeingababy.pmdl"]:
            if self.robot_controller._in_pout_mode:
                print(f"[COMMAND] '{model_name}' detected - exiting pout mode")

                # Special handling for "quit being a baby" - play sassy response first
                if model_name == "quitbeingababy.pmdl":
                    # Disable wake words during response
                    self.robot_controller.input_manager.stop_wake_word_detection()

                    # Play pre-recorded sassy response (in pout pose)
                    await self.robot_controller.play_prerecorded_response(
                        self.robot_controller.quit_baby_response_cache,
                        response_type="QUIT_BABY"
                    )

                    # Re-enable wake words
                    self.robot_controller.input_manager.restart_wake_word_detection()

                    # Fast exit from pout
                    duration = 0.25
                else:
                    # sorrylaura/sorryreachy - gentle exit
                    duration = 2.0

                self.robot_controller.exit_pout_mode(exit_duration=duration)
                self.robot_controller.movement_manager.clear_move_queue()
                self.robot_controller.movement_manager.resume_breathing()
                self.robot_controller.state_tracker.update_state("idle")
                return True # Handled immediately
            else:
                print(f"[COMMAND] '{model_name}' detected but NOT in pout mode - ignoring")
                return False # Not in pout mode, treat as false positive

        # 4. Bluetooth Mode
        if model_name == "tookmycrazypills.pmdl" or model_name == "bluetoothmode.pmdl":
            print("[COMMAND] Bluetooth wake word detected")
            await self.robot_controller.enter_bluetooth_ready()
            return True # Handled immediately

        # 5. Soft Reset
        if model_name == "asyouwere.pmdl":
            print("[COMMAND] 'as you were' detected - resetting to idle")
            if self.robot_controller._in_pout_mode:
                self.robot_controller.exit_pout_mode(exit_duration=2.0)
            self.robot_controller.movement_manager.clear_move_queue()
            self.robot_controller.movement_manager.resume_breathing()
            self.robot_controller.state_tracker.update_state("idle")
            return True

        # 6. Note Transfer (Mac)
        if model_name == "sendnote.pmdl":
            print("[COMMAND] Note transfer wake word detected")
            self.robot_controller.state_tracker.update_state('thinking')
            await self.send_note_to_mac()
            self.robot_controller.state_tracker.update_state('idle')
            return True

        # 7. Send Enter Key
        if model_name == "send_now.pmdl":
            print("[COMMAND] Send Enter wake word detected")
            await self.send_enter_key()
            self.robot_controller.state_tracker.update_state('idle')
            return True

        # 8. Claude Code Flagging (Not handled immediately, but sets flag for InteractionLoop)
        if model_name == "claudecode.pmdl":
            print("[COMMAND] Claude Code wake word detected - setting flag")
            self.claude_code_mode_requested = True
            return False # Proceed to speech capture with flag set
        
        return False # Not a command, proceed with normal interaction

    async def handle_text_command(self, transcript: str) -> bool:
        """
        Handles commands triggered by specific transcribed phrases.
        """
        # Example: "Enter bluetooth mode" voice command
        if "bluetooth mode" in transcript.lower() and "enter" in transcript.lower():
             await self.robot_controller.enter_bluetooth_ready()
             return True
        return False
        
    async def inject_to_claude_code_with_sounds(self, transcript: str):
        """Inject transcript directly into Claude Code terminal using virtual keyboard"""
        try:
            print(f"[INFO] Injecting to Claude Code: '{transcript}'")
            project_root = os.path.dirname(os.path.abspath(__file__))
            # Adjust path to point to root of project relative to this file
            injector_script = os.path.join(project_root, "..", "claude", "claude_voice_injector.py")
            venv_python = os.path.join(project_root, "..", "venv", "bin", "python")
            
            # Copy to clipboard first
            try:
                import pyclip
                pyclip.copy(transcript)
            except Exception:
                pass

            # Play processing sound
            teletype = str(SOUND_BASE_PATH / "sound_effects" / "teletype.mp3")
            if os.path.exists(teletype):
                 await self.robot_controller.audio_coordinator.play_phase_sound("processing", teletype)
            
            # Run injection
            subprocess.run(
                ['sudo', venv_python, injector_script, '--inject-text', transcript],
                capture_output=True, text=True, timeout=30
            )
            
            await self.robot_controller.audio_coordinator.play_phase_sound("complete")
            
        except Exception as e:
            print(f"[ERROR] Failed to inject to Claude Code: {e}")

    async def send_note_to_mac(self):
        """Send pi500_note.txt to Mac server via MCP endpoint"""
        try:
            # Import locally to avoid circular imports or if file is missing
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from send_note_to_mac import Pi500NoteSender
            
            sender = Pi500NoteSender()
            result = sender.send_note()
            
            if result["success"]:
                success_sound = str(SOUND_BASE_PATH / "sound_effects" / "successfulloadup.mp3")
                if os.path.exists(success_sound):
                    await self.robot_controller.audio_coordinator.play_audio_file(success_sound)
                await self.robot_controller.audio_coordinator.tts_handler.speak_text(
                    "Note successfully sent to Mac.",
                    voice_params={"persona": "laura"},
                    coordinator=self.robot_controller.audio_coordinator
                )
            else:
                error_sound = str(SOUND_BASE_PATH / "sound_effects" / "error.mp3")
                if os.path.exists(error_sound):
                    await self.robot_controller.audio_coordinator.play_audio_file(error_sound)
        except Exception as e:
            print(f"[ERROR] Exception in send_note_to_mac: {e}")

    async def send_enter_key(self):
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            enter_script = os.path.join(project_root, "..", "claude", "send_enter.py")
            venv_python = os.path.join(project_root, "..", "venv", "bin", "python")
            subprocess.run(['sudo', venv_python, enter_script], capture_output=True, text=True, timeout=10)
        except Exception as e:
            print(f"[ERROR] Failed to send Enter key: {e}")
