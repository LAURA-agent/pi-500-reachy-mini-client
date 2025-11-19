"""Explicit state machine for robot behavioral states.

Replaces implicit state management with formal state machine semantics.
Backwards compatible with existing StateTracker interface.
"""

import threading
import logging
from typing import Optional, Callable
from statemachine import StateMachine, State

logger = logging.getLogger(__name__)


class RobotStateMachine(StateMachine):
    """Formal state machine for robot conversation and behavior flow.

    States represent distinct behavioral modes.
    Transitions are explicitly defined and enforced.
    Backwards compatible with StateTracker interface for gradual migration.
    """

    # Define states
    sleep = State('sleep', initial=True)
    idle = State('idle')
    listening = State('listening')
    thinking = State('thinking')
    speaking = State('speaking')
    execution = State('execution')
    code = State('code')
    error = State('error')
    pout = State('pout')
    bluetooth_ready = State('bluetooth_ready')
    bluetooth_playing = State('bluetooth_playing')

    # Define transitions
    # Wake transitions
    wake_up = sleep.to(idle)

    # Conversation flow
    wake_word_detected_from_idle = idle.to(listening)
    wake_word_detected_from_pout = pout.to(listening)
    wake_word_detected_from_bluetooth = bluetooth_ready.to(listening)
    speech_captured = listening.to(thinking)
    response_received = thinking.to(speaking)
    speech_complete = speaking.to(idle)

    # Code execution flow
    code_requested = thinking.to(code)
    code_executing = code.to(execution)
    execution_complete = execution.to(idle)

    # Error handling (defined separately for each source state)
    error_from_idle = idle.to(error)
    error_from_listening = listening.to(error)
    error_from_thinking = thinking.to(error)
    error_from_speaking = speaking.to(error)
    error_from_execution = execution.to(error)
    error_from_code = code.to(error)
    error_from_bluetooth_ready = bluetooth_ready.to(error)
    error_from_bluetooth_playing = bluetooth_playing.to(error)
    error_recovered = error.to(idle)

    # Pout mode
    enter_pout = idle.to(pout)
    exit_pout = pout.to(idle)

    # Bluetooth mode
    enter_bluetooth = idle.to(bluetooth_ready)
    bluetooth_audio_detected = bluetooth_ready.to(bluetooth_playing)
    bluetooth_audio_paused = bluetooth_playing.to(bluetooth_ready)
    exit_bluetooth_from_ready = bluetooth_ready.to(idle)
    exit_bluetooth_from_playing = bluetooth_playing.to(idle)

    # Shutdown
    go_to_sleep_from_idle = idle.to(sleep)
    go_to_sleep_from_error = error.to(sleep)

    def __init__(self):
        """Initialize state machine with backwards-compatible interface."""
        super().__init__()

        # Backwards compatibility with StateTracker
        self.display_profile = "laura"  # Current persona: 'laura' or 'claude_code'
        self._state_lock = threading.Lock()
        self._state_callbacks = []

        # Metadata for current state
        self._current_mood = None
        self._current_text = None

        logger.info("RobotStateMachine initialized")

    # Backwards-compatible interface (matches StateTracker)

    def get_state(self) -> str:
        """Get current state (thread-safe, backwards compatible)."""
        with self._state_lock:
            return self.current_state.id

    def update_state(self, state: str, mood: Optional[str] = None, text: Optional[str] = None):
        """Update state with automatic transition selection (backwards compatible).

        Args:
            state: Target state name
            mood: Optional mood indicator
            text: Optional state message
        """
        with self._state_lock:
            old_state = self.current_state.id

            # Store metadata
            self._current_mood = mood
            self._current_text = text

            try:
                # Find and execute appropriate transition
                self._auto_transition_to(state)

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

            except Exception as e:
                logger.error(f"State transition failed: {old_state} → {state}: {e}")
                # Fallback to error state if transition fails
                if state != 'error':
                    # Try to transition to error based on current state
                    error_transitions = {
                        'idle': self.error_from_idle,
                        'listening': self.error_from_listening,
                        'thinking': self.error_from_thinking,
                        'speaking': self.error_from_speaking,
                        'execution': self.error_from_execution,
                        'code': self.error_from_code,
                        'bluetooth_ready': self.error_from_bluetooth_ready,
                        'bluetooth_playing': self.error_from_bluetooth_playing,
                    }
                    if old_state in error_transitions:
                        error_transitions[old_state]()

    def _auto_transition_to(self, target_state: str):
        """Automatically find and execute transition to target state.

        Raises:
            ValueError: If no valid transition exists
        """
        current = self.current_state.id

        # Map target states to transition methods
        transitions_map = {
            'idle': {
                'sleep': self.wake_up,
                'listening': self.speech_complete,
                'speaking': self.speech_complete,
                'thinking': self.speech_complete,
                'execution': self.execution_complete,
                'code': self.execution_complete,
                'error': self.error_recovered,
                'pout': self.exit_pout,
                'bluetooth_ready': self.exit_bluetooth_from_ready,
                'bluetooth_playing': self.exit_bluetooth_from_playing,
            },
            'listening': {
                'idle': self.wake_word_detected_from_idle,
                'pout': self.wake_word_detected_from_pout,
                'bluetooth_ready': self.wake_word_detected_from_bluetooth,
            },
            'thinking': {
                'listening': self.speech_captured,
            },
            'speaking': {
                'thinking': self.response_received,
            },
            'code': {
                'thinking': self.code_requested,
            },
            'execution': {
                'code': self.code_executing,
            },
            'error': {
                # Any state can go to error
                'idle': self.error_from_idle,
                'listening': self.error_from_listening,
                'thinking': self.error_from_thinking,
                'speaking': self.error_from_speaking,
                'execution': self.error_from_execution,
                'code': self.error_from_code,
                'bluetooth_ready': self.error_from_bluetooth_ready,
                'bluetooth_playing': self.error_from_bluetooth_playing,
            },
            'pout': {
                'idle': self.enter_pout,
            },
            'bluetooth_ready': {
                'idle': self.enter_bluetooth,
                'bluetooth_playing': self.bluetooth_audio_paused,
            },
            'bluetooth_playing': {
                'bluetooth_ready': self.bluetooth_audio_detected,
            },
            'sleep': {
                'idle': self.go_to_sleep_from_idle,
                'error': self.go_to_sleep_from_error,
            },
        }

        # If already in target state, no-op
        if current == target_state:
            return

        # Find transition
        if target_state in transitions_map and current in transitions_map[target_state]:
            transition = transitions_map[target_state][current]
            transition()
        else:
            raise ValueError(f"No valid transition from '{current}' to '{target_state}'")

    def register_callback(self, callback: Callable):
        """Register callback for state changes (backwards compatible).

        Callback signature: callback(new_state, old_state, mood, text)
        """
        self._state_callbacks.append(callback)
        logger.debug(f"Registered state callback: {callback.__name__}")

    # Convenience methods (backwards compatible)

    def is_idle(self) -> bool:
        """Check if currently idle."""
        return self.current_state == self.idle

    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self.current_state == self.listening

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.current_state == self.speaking

    def is_thinking(self) -> bool:
        """Check if currently thinking."""
        return self.current_state == self.thinking

    def cleanup(self):
        """Clean up resources (backwards compatible)."""
        logger.info("RobotStateMachine cleanup complete")

    # State machine event handlers (optional - for adding side effects later)

    def on_enter_listening(self):
        """Called when entering listening state."""
        logger.debug("Entered listening state")

    def on_exit_listening(self):
        """Called when exiting listening state."""
        logger.debug("Exited listening state")

    def on_enter_speaking(self):
        """Called when entering speaking state."""
        logger.debug("Entered speaking state")

    def on_exit_speaking(self):
        """Called when exiting speaking state."""
        logger.debug("Exited speaking state")


# Class constants for backwards compatibility
RobotStateMachine.SLEEP = "sleep"
RobotStateMachine.IDLE = "idle"
RobotStateMachine.LISTENING = "listening"
RobotStateMachine.THINKING = "thinking"
RobotStateMachine.SPEAKING = "speaking"
RobotStateMachine.ERROR = "error"
RobotStateMachine.CODE = "code"
RobotStateMachine.EXECUTION = "execution"
RobotStateMachine.BLUETOOTH_READY = "bluetooth_ready"
RobotStateMachine.BLUETOOTH_PLAYING = "bluetooth_playing"

RobotStateMachine.VALID_STATES = {
    RobotStateMachine.SLEEP,
    RobotStateMachine.IDLE,
    RobotStateMachine.LISTENING,
    RobotStateMachine.THINKING,
    RobotStateMachine.SPEAKING,
    RobotStateMachine.ERROR,
    RobotStateMachine.CODE,
    RobotStateMachine.EXECUTION,
    RobotStateMachine.BLUETOOTH_READY,
    RobotStateMachine.BLUETOOTH_PLAYING,
}
