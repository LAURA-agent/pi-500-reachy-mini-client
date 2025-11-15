"""Lightweight state tracker for Reachy conversation flow.

Replaces DisplayManager's pygame/SDL rendering with simple state tracking.
No visual output - just maintains current state for internal logic.
"""

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StateTracker:
    """Track conversation state without visual rendering."""

    # Valid states
    SLEEP = "sleep"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    CODE = "code"
    EXECUTION = "execution"
    BLUETOOTH_READY = "bluetooth_ready"
    BLUETOOTH_PLAYING = "bluetooth_playing"

    VALID_STATES = {SLEEP, IDLE, LISTENING, THINKING, SPEAKING, ERROR, CODE, EXECUTION, BLUETOOTH_READY, BLUETOOTH_PLAYING}

    def __init__(self):
        """Initialize state tracker."""
        self.current_state = self.SLEEP
        self.display_profile = "laura"  # Current persona: 'laura' or 'claude_code'
        self._state_lock = threading.Lock()
        self._state_callbacks = []
        logger.info("StateTracker initialized")

    def update_state(self, state: str, mood: Optional[str] = None, text: Optional[str] = None):
        """Update current state with optional metadata.

        Args:
            state: New state (must be in VALID_STATES)
            mood: Optional mood indicator
            text: Optional state message
        """
        if state not in self.VALID_STATES:
            logger.warning(f"Invalid state '{state}', ignoring")
            return

        with self._state_lock:
            old_state = self.current_state
            self.current_state = state

            # Log state change
            msg = f"State: {old_state} → {state}"
            if mood:
                msg += f" (mood: {mood})"
            if text:
                msg += f" - {text}"
            logger.info(msg)

            # Trigger callbacks
            for callback in self._state_callbacks:
                try:
                    callback(state, old_state, mood, text)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    def get_state(self) -> str:
        """Get current state (thread-safe)."""
        with self._state_lock:
            return self.current_state

    def register_callback(self, callback):
        """Register callback for state changes.

        Callback signature: callback(new_state, old_state, mood, text)
        """
        self._state_callbacks.append(callback)
        logger.debug(f"Registered state callback: {callback.__name__}")

    def is_idle(self) -> bool:
        """Check if currently idle."""
        return self.get_state() == self.IDLE

    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self.get_state() == self.LISTENING

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.get_state() == self.SPEAKING

    def is_thinking(self) -> bool:
        """Check if currently thinking."""
        return self.get_state() == self.THINKING

    def cleanup(self):
        """Clean up resources (placeholder for compatibility)."""
        logger.info("StateTracker cleanup complete")
