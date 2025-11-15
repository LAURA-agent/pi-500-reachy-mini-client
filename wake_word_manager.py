"""Wake Word Manager for context-aware wake word detection and behavior mapping.

This module provides the WakeWordManager class which loads wake word configurations
from YAML and manages which wake words are active based on current robot state.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)


class WakeWordBehavior:
    """Represents a single wake word behavior configuration."""

    def __init__(self, config: Dict):
        """Initialize wake word behavior from config dict.

        Args:
            config: Dictionary from YAML with keys:
                - phrase: Wake word phrase
                - context_injection: System instruction context to add
                - movement_sequence: Name of movement sequence (or null)
                - active_in_states: List of states where this wake word is active
                - scene_manager_override: Whether laura_agent decides in scene mode
        """
        self.phrase = config["phrase"]
        self.context_injection = config["context_injection"]
        self.movement_sequence = config.get("movement_sequence")
        self.active_in_states = config.get("active_in_states", [])
        self.scene_manager_override = config.get("scene_manager_override", False)

    def is_active_in_state(self, current_state: str) -> bool:
        """Check if this wake word is active in the given state.

        Args:
            current_state: Current robot state (idle, pout, sleep, etc.)

        Returns:
            True if wake word should be listened for in this state
        """
        return current_state in self.active_in_states

    def __repr__(self) -> str:
        return f"WakeWordBehavior(phrase='{self.phrase}', states={self.active_in_states})"


class WakeWordManager:
    """Manages contextual wake word behaviors based on robot state."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize wake word manager.

        Args:
            config_path: Path to wake_words.yml config file.
                        Defaults to config/wake_words.yml relative to this file.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "wake_words.yml"

        self.config_path = config_path
        self.behaviors: List[WakeWordBehavior] = []
        self._phrase_to_behavior: Dict[str, WakeWordBehavior] = {}

        self.load_config()

    def load_config(self) -> None:
        """Load wake word configurations from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.behaviors = []
            self._phrase_to_behavior = {}

            for wake_word_config in config.get("wake_words", []):
                behavior = WakeWordBehavior(wake_word_config)
                self.behaviors.append(behavior)
                self._phrase_to_behavior[behavior.phrase.lower()] = behavior

            logger.info(f"Loaded {len(self.behaviors)} wake word behaviors from {self.config_path}")

        except FileNotFoundError:
            logger.error(f"Wake word config file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load wake word config: {e}")
            raise

    def get_active_wake_words(
        self,
        current_state: str,
        scene_manager_active: bool = False
    ) -> List[str]:
        """Get list of wake word phrases active in current state.

        Args:
            current_state: Current robot state (idle, pout, sleep, etc.)
            scene_manager_active: Whether scene_manager scene is currently active

        Returns:
            List of wake word phrases to listen for
        """
        active_phrases = []

        for behavior in self.behaviors:
            if not behavior.is_active_in_state(current_state):
                continue

            # If scene_manager is active and behavior has override flag,
            # include it (laura_agent will decide whether to respond)
            # If scene_manager not active, always include active behaviors
            if scene_manager_active and not behavior.scene_manager_override:
                continue

            active_phrases.append(behavior.phrase)

        logger.debug(f"Active wake words in state '{current_state}': {active_phrases}")
        return active_phrases

    def get_behavior(self, phrase: str) -> Optional[WakeWordBehavior]:
        """Get behavior configuration for a matched wake word phrase.

        Args:
            phrase: Wake word phrase that was detected

        Returns:
            WakeWordBehavior if phrase is known, None otherwise
        """
        return self._phrase_to_behavior.get(phrase.lower())

    def inject_context(self, phrase: str) -> Optional[str]:
        """Get context injection string for a wake word phrase.

        Args:
            phrase: Wake word phrase that was detected

        Returns:
            Context string to append to system instructions, or None
        """
        behavior = self.get_behavior(phrase)
        if behavior:
            return behavior.context_injection
        return None

    def get_movement_sequence_name(self, phrase: str) -> Optional[str]:
        """Get movement sequence name for a wake word phrase.

        Args:
            phrase: Wake word phrase that was detected

        Returns:
            Movement sequence name (e.g., "pout_exit_lunge"), or None
        """
        behavior = self.get_behavior(phrase)
        if behavior:
            return behavior.movement_sequence
        return None

    def requires_scene_manager_approval(self, phrase: str) -> bool:
        """Check if wake word requires scene_manager/laura_agent approval.

        Args:
            phrase: Wake word phrase that was detected

        Returns:
            True if laura_agent should decide whether to respond
        """
        behavior = self.get_behavior(phrase)
        if behavior:
            return behavior.scene_manager_override
        return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    manager = WakeWordManager()

    # Test: What wake words are active when pouting?
    print("\n=== Pouting State ===")
    pout_wake_words = manager.get_active_wake_words("pout", scene_manager_active=False)
    print(f"Active wake words: {pout_wake_words}")

    # Test: Get behavior for specific phrase
    print("\n=== 'quit being a baby laura' Behavior ===")
    phrase = "quit being a baby laura"
    behavior = manager.get_behavior(phrase)
    if behavior:
        print(f"Phrase: {behavior.phrase}")
        print(f"Context: {behavior.context_injection[:100]}...")
        print(f"Movement: {behavior.movement_sequence}")
        print(f"Scene override: {behavior.scene_manager_override}")

    # Test: Idle state wake words
    print("\n=== Idle State ===")
    idle_wake_words = manager.get_active_wake_words("idle", scene_manager_active=False)
    print(f"Active wake words: {idle_wake_words}")
