# core/lifecycle_manager.py
"""
Manages the startup and shutdown of various application components,
ensuring a clean and ordered lifecycle.
"""
import asyncio

class LifecycleManager:
    def __init__(self, robot_controller, api_server, audio_manager, 
                 state_tracker, input_manager):
        self.robot_controller = robot_controller
        self.api_server = api_server
        self.audio_manager = audio_manager
        self.state_tracker = state_tracker
        self.input_manager = input_manager
        self._api_server_runner = None
        self._background_tasks = []

    async def startup(self):
        """Initializes and starts all necessary background services."""
        print("[INFO] --- Application Startup Sequence ---")

        self._api_server_runner = await self.api_server.start()
        print("[INFO] API Server started.")

        # Only start audio input stream if method exists (hardware mode)
        if hasattr(self.audio_manager, 'start_input_stream'):
            await self.audio_manager.start_input_stream()
            print("[INFO] Audio stream started.")

        self.robot_controller.start_threads()
        print("[INFO] Robot control threads started.")

        # Only start sleep monitor if input_manager exists
        if self.input_manager:
            sleep_monitor = asyncio.create_task(self.sleep_timeout_monitor())
            self._background_tasks.append(sleep_monitor)

        self.state_tracker.update_state("idle")
        print("[INFO] Application is idle and ready.")

    async def shutdown(self):
        """Gracefully stops all services and cleans up resources."""
        print("[INFO] --- Application Shutdown Sequence ---")

        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        print("[INFO] Background tasks cancelled.")

        self.robot_controller.stop_threads()
        print("[INFO] Robot control threads stopped.")

        if self._api_server_runner:
            await self._api_server_runner.cleanup()
            print("[INFO] API Server stopped.")

        if self.audio_manager:
            await self.audio_manager.cleanup()
        if self.state_tracker:
            self.state_tracker.cleanup()
        if self.input_manager:
            self.input_manager.cleanup()
        
        print("[INFO] All resources cleaned up.")
        
    async def sleep_timeout_monitor(self):
        """Background task to monitor for inactivity and trigger sleep mode."""
        SLEEP_TIMEOUT = 300  # 5 minutes
        while True:
            try:
                if self.state_tracker.get_state() == 'idle':
                    time_since_interaction = self.input_manager.get_time_since_last_interaction()
                    if time_since_interaction >= SLEEP_TIMEOUT:
                        print(f"[INFO] Sleep timeout reached ({time_since_interaction:.1f}s). Entering sleep mode.")
                        self.robot_controller.enter_sleep_mode()
                await asyncio.sleep(5) # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ERROR] Sleep timeout monitor error: {e}")
                await asyncio.sleep(15)
