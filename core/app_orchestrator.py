# core/app_orchestrator.py
"""
The central orchestrator for the application.
Initializes all manager classes, links them, and handles the main
server connection loop, including the offline "Code Mode" fallback.
"""

import asyncio
import traceback
from colorama import Fore, Style, init

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from config.client_config import SERVER_URL, DEVICE_ID, client_settings, save_client_settings, get_active_tts_provider, set_active_tts_provider
from hardware.robot_controller import RobotController
from communication.mcp_session_manager import MCPSessionManager
from communication.api_server import APIServer
from core.interaction_loop import InteractionLoop
from core.lifecycle_manager import LifecycleManager
from system.audio_manager import AudioManager
from display.display_manager import DisplayManager
from robot_state_machine import RobotStateMachine
from system.input_manager import InputManager
from system.notification_manager import NotificationManager
from system.conversation_manager import ConversationManager
from speech_capture.speech_processor import SpeechProcessor
from speech_capture.vosk_websocket_adapter import VoskTranscriber
from communication.client_tts_handler import TTSHandler
from system.system_command_manager import SystemCommandManager
from utils.helpers import SOUND_BASE_PATH
import os

init()

class AppOrchestrator:
    def __init__(self):
        # Initialize state machine and display separately
        self.state_machine = RobotStateMachine()
        self.display_manager = DisplayManager()

        # Connect state machine to display manager via callback
        self.state_machine.register_callback(self._on_state_change_for_display)

        self.audio_manager = AudioManager()
        self.tts_handler = TTSHandler()
        self.input_manager = InputManager(self.audio_manager)
        self.robot_controller = RobotController(self.state_machine, self.input_manager)
        self.transcriber = VoskTranscriber()
        self.speech_processor = SpeechProcessor(
            self.audio_manager, self.transcriber, None
        )
        self.conversation_manager = ConversationManager(
            self.speech_processor, self.robot_controller.audio_coordinator, self.tts_handler,
            client_settings, self.robot_controller.movement_manager
        )
        self.mcp_session_manager = MCPSessionManager()
        self.api_server = APIServer(self.robot_controller, self.state_machine, self.audio_manager, self.input_manager)
        self.notification_manager = NotificationManager(
            self.robot_controller.audio_coordinator, self.tts_handler
        )
        self.system_command_manager = SystemCommandManager(
            client_settings, save_client_settings, get_active_tts_provider, set_active_tts_provider
        )
        self.system_command_manager._notification_manager = self.notification_manager

        self.interaction_loop = InteractionLoop(
            state_tracker=self.state_machine, input_manager=self.input_manager,
            robot_controller=self.robot_controller, mcp_session_manager=self.mcp_session_manager,
            speech_processor=self.speech_processor, conversation_manager=self.conversation_manager,
            system_command_manager=self.system_command_manager
        )
        self.lifecycle_manager = LifecycleManager(
            robot_controller=self.robot_controller, api_server=self.api_server,
            audio_manager=self.audio_manager, state_tracker=self.state_machine,
            input_manager=self.input_manager
        )
        self.input_manager.initialize_keyboard()
        self.speech_processor.keyboard_device = self.input_manager.keyboard_device

    def _on_state_change_for_display(self, new_state: str, old_state: str, mood: str = None, text: str = None):
        """Callback to update display when state machine changes state."""
        self.display_manager.update_display_sync(new_state, mood, text)

    async def run(self):
        await self.lifecycle_manager.startup()
        
        connection_failures = 0
        code_mode_active = False

        while True:
            try:
                if code_mode_active:
                    print(f"{Fore.YELLOW}[WARNING] Running in OFFLINE CODE MODE - server connection bypassed{Style.RESET_ALL}")
                    self.state_machine.update_state("idle")
                    self.interaction_loop.set_offline_mode(True)
                    # The loop will run accepting keyboard/voice commands for local injection
                    await self.interaction_loop.run()
                    # If the loop somehow exits, wait before attempting to reconnect
                    await asyncio.sleep(10)
                    continue

                print(f"[INFO] Attempting to connect to MCP server at {SERVER_URL}...")
                async with sse_client(f"{SERVER_URL}/events/sse", headers={}) as (read, write):
                    async with ClientSession(read, write) as session:
                        print("[INFO] SSE Connection successful.")
                        self.mcp_session_manager.set_session(session)
                        
                        if not await self.mcp_session_manager.initialize_session(DEVICE_ID):
                            raise ConnectionAbortedError("Failed to initialize MCP session.")

                        # On successful connection, reset failures and exit offline mode
                        connection_failures = 0
                        code_mode_active = False
                        self.interaction_loop.set_offline_mode(False)

                        self.state_machine.update_state("idle")
                        print(f"{Fore.GREEN}✓ Session initialized successfully{Style.RESET_ALL}")
                        self.interaction_loop.set_session_id(self.mcp_session_manager.session_id)

                        # Play startup sound on successful connection
                        try:
                            from pathlib import Path
                            startup_sound = Path("/home/user/reachy/pi_reachy_deployment/assets/sounds/sound_effects/successfulloadup.mp3")
                            if startup_sound.exists():
                                print(f"{Fore.CYAN}[INFO] Playing startup audio...{Style.RESET_ALL}")
                                await self.robot_controller.audio_coordinator.play_audio_file(str(startup_sound))
                                print(f"{Fore.GREEN}✓ Startup audio complete{Style.RESET_ALL}")
                        except Exception as e:
                            print(f"{Fore.YELLOW}[WARNING] Could not play startup sound: {e}{Style.RESET_ALL}")

                        # TODO: Start notification background task
                        # notification_task = asyncio.create_task(
                        #     self.notification_manager.check_for_notifications_loop(
                        #         self.mcp_session_manager.mcp_session,
                        #         self.mcp_session_manager.session_id,
                        #         self.state_machine
                        #     )
                        # )

                        await self.interaction_loop.run()

            except (ConnectionRefusedError, ConnectionError, OSError, ConnectionAbortedError) as e:
                print(f"{Fore.RED}[ERROR] Connection failed: {e}.{Style.RESET_ALL}")
                connection_failures += 1

                if connection_failures >= 2 and not code_mode_active:
                    print(f"{Fore.YELLOW}[WARNING] {connection_failures} connection failures. Entering OFFLINE CODE MODE.{Style.RESET_ALL}")
                    code_mode_active = True

                if not code_mode_active:
                    print(f"[INFO] Retrying connection in 30 seconds...")
                    await asyncio.sleep(30)

            except asyncio.CancelledError:
                print(f"{Fore.YELLOW}[INFO] Application cancelled - shutting down.{Style.RESET_ALL}")
                raise
            except KeyboardInterrupt:
                print(f"{Fore.YELLOW}[INFO] KeyboardInterrupt received - shutting down.{Style.RESET_ALL}")
                raise
            except Exception as e:
                # Check if this is an MCP connection error (RemoteProtocolError, httpx errors, etc.)
                error_name = type(e).__name__
                is_mcp_error = any(keyword in error_name.lower() for keyword in ['remote', 'protocol', 'httpx', 'sse'])

                if is_mcp_error:
                    print(f"{Fore.RED}[ERROR] MCP server connection failed: {error_name}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}[INFO] Server at {SERVER_URL} is offline or unreachable.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}[ERROR] Unhandled connection-level exception: {e}{Style.RESET_ALL}")
                    traceback.print_exc()

                connection_failures += 1
                if connection_failures >= 2 and not code_mode_active:
                    print(f"{Fore.YELLOW}[WARNING] {connection_failures} connection failures. Entering OFFLINE CODE MODE.{Style.RESET_ALL}")
                    code_mode_active = True
                if not code_mode_active:
                    await asyncio.sleep(30)
            finally:
                self.mcp_session_manager.clear_session()

    async def cleanup(self):
        await self.lifecycle_manager.shutdown()
