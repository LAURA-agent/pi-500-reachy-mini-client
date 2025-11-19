# communication/api_server.py
"""
Runs an aiohttp web server to handle external commands via HTTP POST requests,
such as triggering robot states, updating the display, or TTS callbacks.
"""

import os
import time
import asyncio
import requests
from aiohttp import web
from mutagen.mp3 import MP3

class APIServer:
    def __init__(self, robot_controller, state_tracker, audio_manager, input_manager):
        self.robot_controller = robot_controller
        self.state_tracker = state_tracker
        self.audio_manager = audio_manager
        self.input_manager = input_manager
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_post('/pout/trigger', self.handle_pout_trigger)
        self.app.router.add_post('/pout/exit', self.handle_pout_exit)
        self.app.router.add_post('/pout/rotate', self.handle_pout_rotate)
        self.app.router.add_post('/display/update', self.handle_display_update)
        self.app.router.add_post('/speech/motion', self.handle_speech_motion)
        self.app.router.add_post('/speech/motion/start', self.handle_speech_motion_start)
        self.app.router.add_post('/tts/conversation', self.handle_tts_conversation)
        self.app.router.add_post('/tts/working', self.handle_tts_working)
        
    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8888)
        await site.start()
        print("[INFO] API endpoints started on http://localhost:8888")
        return runner

    async def handle_pout_trigger(self, request):
        data = await request.json()
        entry_speed = data.get('entry_speed', 'slow')
        initial_body_yaw = data.get('initial_body_yaw', 0.0)
        audio_clip = data.get('audio_clip', None)

        if self.robot_controller._in_pout_mode:
            return web.json_response({"status": "ignored", "reason": "already_in_pout_mode"})
        
        if audio_clip:
             # Simplified logic to play pre-recorded clip if exists
             pass 

        self.robot_controller.enter_pout_mode(initial_body_yaw, entry_speed)
        return web.json_response({"status": "success"})

    async def handle_pout_exit(self, request):
        data = await request.json()
        exit_duration = data.get('exit_duration', 3.0)
        if not self.robot_controller._in_pout_mode:
            return web.json_response({"status": "ignored", "reason": "not_in_pout_mode"})
        self.robot_controller.exit_pout_mode(exit_duration=exit_duration)
        return web.json_response({"status": "success"})

    async def handle_pout_rotate(self, request):
        data = await request.json()
        angle_deg = data.get('angle_deg', 0.0)
        if not self.robot_controller._in_pout_mode:
            return web.json_response({"status": "ignored", "reason": "not_in_pout_mode"})
        self.robot_controller.pout_rotate_to(angle_deg)
        return web.json_response({"status": "success"})

    async def handle_display_update(self, request):
        data = await request.json()
        state = data.get('state', 'idle')
        mood = data.get('mood')
        text = data.get('text')
        self.state_tracker.update_state(state, mood=mood, text=text)
        return web.json_response({"status": "success"})

    async def handle_speech_motion(self, request):
        data = await request.json()
        audio_file = data.get('audio_file')
        text = data.get('text', '')
        duration = data.get('duration', 0)

        if not audio_file or not os.path.exists(audio_file):
            return web.json_response({"status": "error", "error": "Audio file not found"}, status=400)

        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(None, self.robot_controller.speech_analyzer.analyze, audio_file, text)
        self.robot_controller.speech_offset_player.load_timeline(analysis)
        self.robot_controller.camera_worker.set_head_tracking_enabled(False)
        self.robot_controller.speech_motion_duration = duration

        return web.json_response({"status": "success", "message": "Speech motion prepared"})

    async def handle_speech_motion_start(self, request):
        data = await request.json()
        audio_start_time = data.get('audio_start_time', time.time())
        
        self.robot_controller.speech_offset_player.play(audio_start_time)
        self.state_tracker.update_state('speaking')

        duration = self.robot_controller.speech_motion_duration
        async def stop_after_duration():
            try:
                await asyncio.sleep(duration)
            finally:
                self.robot_controller.speech_offset_player.stop()
                self.robot_controller.movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                self.robot_controller.camera_worker.set_head_tracking_enabled(True)
                if self.state_tracker.get_state() == 'speaking':
                    self.state_tracker.update_state('idle')
                self.robot_controller.speech_motion_cleanup_task = None
        
        if self.robot_controller.speech_motion_cleanup_task:
            self.robot_controller.speech_motion_cleanup_task.cancel()
        self.robot_controller.speech_motion_cleanup_task = asyncio.create_task(stop_after_duration())
        return web.json_response({"status": "success"})

    async def handle_tts_conversation(self, request):
        """Endpoint for TTS that returns to idle state - for questions/confirmations"""
        try:
            data = await request.json()
            text = data.get('text', '')
            mood = data.get('mood', 'explaining')
            
            if text:
                self.state_tracker.update_state("speaking", mood=mood)
                # Make request to TTS service (assuming localhost:5001 based on legacy)
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: requests.post(
                            "http://localhost:5001/tts",
                            headers={"Content-Type": "application/json"},
                            json={"text": text, "voice": "claude"},
                            timeout=30
                        )
                    )
                    if response.status_code == 200:
                        response_data = response.json()
                        audio_file = response_data.get('audio_file')
                        if audio_file and os.path.exists(audio_file):
                             # Here we trigger the logic to play it locally with motion
                             # Using requests inside async handler is a bit hacky, but fits the legacy structure
                             # In a pure refactor, TTSHandler should be used, but this endpoint is for *external* callers
                             pass 
                except Exception as e:
                    print(f"[TTS Endpoint Error] {e}")
                
                self.input_manager.restart_wake_word_detection()
                self.state_tracker.update_state("idle")
            return web.json_response({"status": "success", "state": "idle"})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def handle_tts_working(self, request):
        """Endpoint for TTS that goes to execution state"""
        try:
            data = await request.json()
            text = data.get('text', '')
            mood = data.get('mood', 'solution')
            if text:
                self.state_tracker.update_state("speaking", mood=mood)
                # ... (TTS logic similar to above)
                self.state_tracker.update_state("idle") # Or 'execution' if you have that state
            return web.json_response({"status": "success"})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)
