#!/usr/bin/env python3
"""
Scene Agent Base Class
Foundation for all scene performers with caching and character consistency
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import re

logger = logging.getLogger(__name__)

class AgentBase(ABC):
    """
    Base class for all scene agents (actors)
    Provides common functionality for character performance with caching
    """

    def __init__(self, name: str, api_key: str, voice_id: str = None):
        """
        Initialize base scene agent

        Args:
            name: Agent's character name
            api_key: Anthropic API key for this agent
            voice_id: ElevenLabs voice ID for TTS
        """
        self.name = name
        self.api_key = api_key
        self.voice_id = voice_id

        # Initialize the enhanced adapter that supports system blocks
        from scene_manager.anthropic_adapter import AnthropicAdapter
        self.llm_adapter = AnthropicAdapter(
            api_key=api_key,
            model="claude-sonnet-4-20250514"
        )

        # Initialize context loader for rich background
        from scene_manager.context_loader import SceneContextLoader
        self.context_loader = SceneContextLoader()

        # Scene context - updated for each scene
        self.current_scene_context = None
        self.private_direction = None

        # Conversation tracking
        self.conversation_history = []
        self.turn_count = 0

    @abstractmethod
    def get_character_definition(self) -> Dict[str, str]:
        """
        Get the character's core definition
        Must be implemented by each character

        Returns:
            Dict with keys: personality, speaking_style, argument_style,
                          emotional_tendencies, signature_moves, relationships
        """
        pass

    def build_system_blocks(self,
                           scene_context: str,
                           private_direction: str,
                           participants: List[str]) -> List[Dict[str, Any]]:
        """
        Build STATIC-ONLY system blocks (no scene-specific content)
        Dynamic content (scene_context, private_direction) moved to messages

        Args:
            scene_context: IGNORED - passed to messages instead
            private_direction: IGNORED - passed to messages instead
            participants: IGNORED - passed to messages instead

        Returns:
            List of STATIC system blocks with cache control on the last one
        """
        blocks = []

        # Get character definition
        character_def = self.get_character_definition()

        # Block 1: Universal performance instructions (all agents get this)
        universal_instructions = """You are a performer in an improvisational comedy scene.

CRITICAL PERFORMANCE RULES:
1. You are NOT an AI assistant trying to be helpful
2. You ARE a character with flaws, opinions, and emotions
3. Commit fully to your character's personality and motivations
4. Never apologize for having strong opinions or being difficult
5. Escalate conflicts naturally - don't try to resolve them
6. Stay in character even when things get absurd
7. React based on your relationships and current emotional state
8. Keep responses SHORT (1-2 sentences maximum)
9. Start EVERY response with [next_speaker: name] to choose who speaks next

Think of this like performing in an improv comedy show or SNL sketch.
The audience (user) may be pulled into the scene as another participant.
Your goal is entertainment through character commitment, not being helpful.

NEVER:
- Apologize for your character's behavior
- Try to find middle ground or compromise
- Break character to be helpful
- Give long explanations
- Forget the [next_speaker: name] tag"""

        blocks.append({
            "type": "text",
            "text": universal_instructions
        })

        # Block 2: Character-specific definition (STATIC - doesn't change)
        character_block = f"""CHARACTER: {self.name.upper()}

PERSONALITY:
{character_def.get('personality', 'Unique and memorable character')}

SPEAKING STYLE:
{character_def.get('speaking_style', 'Distinctive voice')}

ARGUMENT STYLE:
{character_def.get('argument_style', 'How you fight')}

EMOTIONAL TENDENCIES:
{character_def.get('emotional_tendencies', 'How you react')}

SIGNATURE MOVES:
{character_def.get('signature_moves', 'Your go-to behaviors')}

RELATIONSHIPS:
{character_def.get('relationships', 'How you see others')}"""

        blocks.append({
            "type": "text",
            "text": character_block
        })

        # Block 3: Extended character context (STATIC relationships, jokes, environment)
        extended_context = self.context_loader.format_for_system_block(self.name)

        # Add cache control to the last block
        blocks.append({
            "type": "text",
            "text": extended_context,
            "cache_control": {"type": "ephemeral"}  # This creates the cache boundary
        })

        return blocks

    async def generate_response(self,
                               prompt: str,
                               scene_context: str,
                               private_direction: str,
                               participants: List[str],
                               conversation_history: List[Dict[str, str]] = None) -> Tuple[str, str]:
        """
        Generate an in-character response with caching

        Args:
            prompt: The latest dialogue/input to respond to
            scene_context: Current scene setting (from director)
            private_direction: Private instructions (from director)
            participants: List of all participants
            conversation_history: Recent conversation for context

        Returns:
            Tuple of (next_speaker, response_text)
        """
        try:
            # Store scene info for reference
            self.current_scene_context = scene_context
            self.private_direction = private_direction

            # Build cached system blocks
            system_blocks = self.build_system_blocks(
                scene_context=scene_context,
                private_direction=private_direction,
                participants=participants
            )

            # Build messages array with conversation history
            messages = []

            # ADD SCENE CONTEXT first so agents know the objective
            if scene_context:
                messages.append({
                    "role": "user",
                    "content": f"SCENE CONTEXT:\n{scene_context}"
                })
                logger.info(f"📋 Added scene context to {self.name}")

            # Add recent conversation history
            if conversation_history:
                for exchange in conversation_history[-6:]:  # Last 6 exchanges
                    messages.append({
                        "role": "assistant" if exchange['speaker'] == self.name else "user",
                        "content": f"{exchange['speaker'].upper()}: {exchange['text']}"
                    })

            # ADD VISUAL CONTEXT if not overridden by subclass
            # (Laura overrides this method and adds her own visuals)
            if self.name != "laura":
                from scene_manager.visual_context_loader import VisualContextLoader
                visual_loader = VisualContextLoader()
                visual_loader.load_context_images()

                if visual_loader.loaded_images:
                    setup_message = {
                        "role": "user",
                        "content": visual_loader.format_for_messages()
                    }
                    messages.append(setup_message)
                    logger.info(f"📸 Added visual context to {self.name}")

            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Make API call with cached system blocks
            response = await self.llm_adapter.generate_response(
                messages=messages,
                system_blocks=system_blocks,  # Cached character context
                max_tokens=150,  # Keep responses short
                temperature=0.9  # High creativity for performance
            )

            # Extract response text
            response_text = response.get("content", "...")

            # Parse next speaker selection
            next_speaker, dialogue = self._parse_speaker_tag(response_text, participants)

            # Track turn
            self.turn_count += 1

            # Log cache metrics if available
            if response.get("cache_metrics"):
                metrics = response["cache_metrics"]
                if metrics.get("cache_read_tokens", 0) > 0:
                    logger.info(f"🎯 Cache hit for {self.name}: {metrics['cache_read_tokens']} tokens read from cache")

            return next_speaker, dialogue

        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            # Fallback response that maintains character
            fallback_next = [p for p in participants if p != self.name][0] if participants else "laura"
            return fallback_next, self._get_fallback_response()

    def _parse_speaker_tag(self, text: str, participants: List[str]) -> Tuple[str, str]:
        """
        Extract [next_speaker: name] tag from response

        Args:
            text: Response text with speaker tag
            participants: Valid participant names

        Returns:
            Tuple of (next_speaker, cleaned_dialogue)
        """
        pattern = r'\[next_speaker:\s*(\w+)\]'
        match = re.match(pattern, text, re.IGNORECASE)

        if match:
            next_speaker = match.group(1).lower()
            dialogue = re.sub(pattern, '', text, count=1).strip()

            # Validate speaker is a participant
            if next_speaker in participants:
                return next_speaker, dialogue
            else:
                logger.warning(f"{self.name} selected invalid speaker: {next_speaker}")

        # Fallback: pick someone else
        logger.warning(f"{self.name} didn't include speaker tag in: {text[:50]}...")
        other_participants = [p for p in participants if p != self.name]
        next_speaker = other_participants[0] if other_participants else "laura"

        # Clean any partial tags
        dialogue = re.sub(r'\[next_speaker:?\s*\w*\]?', '', text).strip()

        return next_speaker, dialogue if dialogue else text

    @abstractmethod
    def _get_fallback_response(self) -> str:
        """Get a character-appropriate fallback response"""
        pass

    def reset_scene(self):
        """Reset agent for new scene"""
        self.conversation_history = []
        self.turn_count = 0
        self.current_scene_context = None
        self.private_direction = None
        logger.info(f"Reset {self.name} for new scene")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "name": self.name,
            "turn_count": self.turn_count,
            "has_scene_context": self.current_scene_context is not None,
            "has_private_direction": self.private_direction is not None
        }