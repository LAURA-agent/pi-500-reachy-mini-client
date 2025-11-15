#!/usr/bin/env python3

import asyncio
import json
import os
import random
from typing import Optional


class NotificationManager:
    """
    Manages notification checking, processing, and display coordination.
    
    Handles server notification polling, notification audio/TTS coordination,
    and proper display state management during notification events.
    """
    
    def __init__(self, audio_coordinator, tts_handler):
        self.audio_coordinator = audio_coordinator
        self.tts_handler = tts_handler

    async def check_for_notifications(self, mcp_session, session_id):
        """Check for server notifications once"""
        if not mcp_session or not session_id:
            return
        
        try:
            response = await mcp_session.call_tool("check_notifications", 
                                                  arguments={"session_id": session_id})
            
            # Parse response properly
            notifications = None
            if hasattr(response, 'content') and response.content:
                json_str = response.content[0].text
                parsed_response = json.loads(json_str)
                notifications = parsed_response.get("notifications", [])
            elif isinstance(response, dict):
                notifications = response.get("notifications", [])
            
            if notifications:
                return notifications
                    
        except Exception as e:
            # Silently fail - don't spam logs for normal "no notifications" cases
            pass
        
        return []

    async def handle_notification(self, notification, display_manager):
        """Handle incoming notification with TTS and display"""
        notification_type = notification.get("notification_type", "general")
        text = notification.get("text", "")
        minutes_late = notification.get("minutes_late", 0)
        
        print(f"[NOTIFICATION] {notification_type}: {text} (late: {minutes_late}min)")
        
        # Interrupt current activity for notifications
        current_state = display_manager.current_state
        
        # Stop any current audio
        await self.audio_coordinator.stop_current_audio()
        
        # Determine mood and sound for notification
        mood = notification.get("mood", "caring")
        notification_audio = None
        
        # Update display to notification state with appropriate mood
        await display_manager.update_display("notification", mood=mood)
        
        # Play notification sound if available
        if notification_audio and os.path.exists(notification_audio):
            await self.audio_coordinator.play_audio_file(notification_audio)
            await asyncio.sleep(1)  # Brief pause between notification sound and TTS
        
        # Generate and play TTS (display state set to notification with mood)
        if text:
            audio_bytes, engine = await self.tts_handler.generate_audio(text, persona_name="laura")
            if audio_bytes:
                await self.audio_coordinator.handle_tts_playback(audio_bytes, engine)
        
        # Return to previous state or idle
        await self.audio_coordinator.wait_for_audio_completion_with_buffer()
        if current_state in ['sleep', 'idle']:
            await display_manager.update_display(current_state)
        else:
            await display_manager.update_display("idle")

    async def process_notifications(self, notifications, display_manager):
        """Process a list of notifications"""
        for notification in notifications:
            await self.handle_notification(notification, display_manager)

    async def check_for_notifications_loop(self, mcp_session, session_id, display_manager):
        """Background notification checking every 30 seconds"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            notifications = await self.check_for_notifications(mcp_session, session_id)
            if notifications:
                await self.process_notifications(notifications, display_manager)