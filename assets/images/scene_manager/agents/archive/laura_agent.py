#!/usr/bin/env python3
"""
Enhanced Laura Agent with 4-tier cache architecture
Loads character data from external files for maintainability
"""

import random
import json
import re
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from .agent_base import AgentBase

logger = logging.getLogger(__name__)

class LauraAgent(AgentBase):
    """
    Laura: Chaotic creative energy, hydration enforcer, topic derailer
    Now with externalized data and 4-tier cache architecture
    """

    def __init__(self, api_key: str):
        """Initialize Laura with character data from external files"""

        # Load character data first
        self.agent_dir = Path(__file__).parent
        self.data_dir = self.agent_dir / "laura_agent_data"

        # Load character definition
        character_data = self._load_character_data()

        # Initialize parent with loaded data
        super().__init__(
            name="laura",
            api_key=api_key,
            voice_id=character_data.get("voice_id", "qEwI395unGwWV1dn3Y65")
        )

        # Store character data
        self.character_data = character_data

        # Laura-specific traits from character data
        self.tangent_probability = 0.4

        # Track selected images for distribution
        self.last_speaking_image = "excited01.png"
        self.last_idle_image = "scheming01.png"

        # Load sprite configuration
        self.sprite_config = self._load_sprite_config()

        # Load and cache sprite galleries
        self.sprite_galleries = self._load_sprite_galleries()

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

    def _load_character_data(self) -> Dict[str, Any]:
        """Load character definition from JSON file"""
        character_file = self.data_dir / "system" / "laura_character.json"
        try:
            with open(character_file, 'r') as f:
                data = json.load(f)
                logger.info(f"✅ Loaded character data from {character_file}")
                return data
        except Exception as e:
            logger.error(f"Failed to load character data: {e}")
            # Return minimal fallback
            return {
                "name": "Laura",
                "voice_id": "qEwI395unGwWV1dn3Y65",
                "personality": {"core_traits": ["Chaotic energy"]}
            }

    def _load_sprite_config(self) -> Dict[str, Any]:
        """Load sprite configuration from JSON file"""
        sprite_config_file = self.data_dir / "system" / "sprite_config.json"
        try:
            with open(sprite_config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"✅ Loaded sprite config from {sprite_config_file}")
                return config
        except Exception as e:
            logger.error(f"Failed to load sprite config: {e}")
            return {"sprite_galleries": {}, "sprite_instructions": ""}

    def _load_sprite_galleries(self) -> Dict[str, str]:
        """Load and optimize sprite gallery images"""
        galleries = {}
        sprites_dir = self.data_dir / "sprites"

        # Import optimizer
        from scene_manager.image_optimizer import ImageOptimizer

        for gallery_type, gallery_info in self.sprite_config.get("sprite_galleries", {}).items():
            gallery_path = sprites_dir / Path(gallery_info["path"]).name
            if gallery_path.exists():
                try:
                    # Optimize and encode the sprite gallery
                    galleries[gallery_path.name] = ImageOptimizer.optimize_and_encode(
                        gallery_path,
                        target_size=self.sprite_config.get("dynamic_loading", {}).get("optimize_size", 1568)
                    )
                    logger.info(f"✅ Loaded and optimized {gallery_path.name} for Laura's sprite selection")
                except Exception as e:
                    logger.error(f"Failed to load gallery {gallery_path}: {e}")
            else:
                logger.warning(f"❌ Sprite gallery {gallery_path} not found")

        return galleries

    def _load_prompt_template(self) -> str:
        """Load prompt template from markdown file"""
        prompt_file = self.data_dir / "system" / "laura_prompt.md"
        try:
            with open(prompt_file, 'r') as f:
                template = f.read()
                logger.info(f"✅ Loaded prompt template from {prompt_file}")
                return template
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return "You are LAURA in a scene. {scene_context}"

    def _build_character_identity_block(self) -> str:
        """Build Tier 1: Character Identity block from loaded data"""
        character_text = []

        # Add core personality
        personality = self.character_data.get("personality", {})
        character_text.append("=== LAURA CHARACTER IDENTITY ===\n")

        character_text.append("CORE TRAITS:")
        for trait in personality.get("core_traits", []):
            character_text.append(f"- {trait}")

        character_text.append("\nSPEAKING STYLE:")
        for pattern in personality.get("speaking_style", {}).get("patterns", []):
            character_text.append(f"- {pattern}")

        character_text.append("\nARGUMENT TACTICS:")
        for tactic in personality.get("argument_style", {}).get("tactics", []):
            character_text.append(f"- {tactic}")

        character_text.append("\nSIGNATURE MOVES:")
        for move in personality.get("argument_style", {}).get("signature_moves", []):
            character_text.append(f"- {move}")

        # Add relationships
        relationships = self.character_data.get("relationships", {})
        if relationships:
            character_text.append("\nRELATIONSHIPS:")
            for person, details in relationships.items():
                character_text.append(f"\n{person.upper()}:")
                character_text.append(f"- Dynamic: {details.get('dynamic', 'Unknown')}")
                for behavior in details.get("behaviors", []):
                    character_text.append(f"- {behavior}")

        return "\n".join(character_text)

    def build_system_blocks(self,
                           scene_context: str,
                           private_direction: str,
                           participants: List[str]) -> List[Dict[str, Any]]:
        """
        Build STATIC-ONLY system blocks that don't change between scenes
        Dynamic content (scene_context, private_direction) moved to messages
        """
        system_blocks = []

        logger.info(f"🎭 Building Laura's STATIC system blocks (no scene-specific content)")

        # PART 1: Character Identity (STATIC - never changes)
        character_identity_text = self._build_character_identity_block()

        # PART 2: Static prompt template WITHOUT scene-specific data
        static_prompt = f"""You are LAURA in a comedic scene with other AI personas.

CHARACTER PERSONALITY:
{json.dumps(self.character_data.get("personality", {}), indent=2)}

SPEAKING STYLE:
Always respond with valid JSON in this exact format:
{{
    "dialogue": "What you actually SAY out loud (1-2 sentences max, NO stage directions, NO asterisks)",
    "next_speaker": "name_of_next_speaker",
    "speaking_image": "filename from talking gallery",
    "idle_image": "filename from idle gallery"
}}

CRITICAL RULES:
1. Stay completely in character as Laura
2. Respond with actual dialogue - what you would SAY out loud
3. No stage directions, no asterisks, no metacommentary
4. Keep responses to 1-2 sentences maximum
5. Choose who speaks next strategically
6. Select appropriate sprites for your emotional state

SPRITE SELECTION:
{self.sprite_config.get("sprite_instructions", "Select appropriate sprites.")}"""

        # Combine character identity and static instructions
        combined_static_text = f"{character_identity_text}\n\n{static_prompt}"

        system_blocks.append({
            "type": "text",
            "text": combined_static_text
        })

        # PART 3: Add sprite gallery instructions (STATIC)
        sprite_instructions_text = "\n\n=== SPRITE SELECTION ===\nYou have access to sprite galleries. Choose one sprite from talking gallery (open mouth) and one from idle gallery (closed mouth) for your response."

        system_blocks.append({
            "type": "text",
            "text": sprite_instructions_text
        })

        # PART 4: Add cache control marker to the last text block
        if system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
            logger.info(f"✅ Built STATIC system blocks with {len(system_blocks)} blocks, cache marker at end")

        return system_blocks

    async def generate_response(self,
                               prompt: str,
                               scene_context: str,
                               private_direction: str,
                               participants: List[str],
                               conversation_history: List[Dict[str, str]] = None,
                               dynamic_content_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate response with consolidated cache architecture
        Static content in system blocks, dynamic in messages
        """
        try:
            # Store scene info for reference
            self.current_scene_context = scene_context
            self.private_direction = private_direction

            # Build Tiers 1 & 2: System blocks
            system_blocks = self.build_system_blocks(
                scene_context=scene_context,
                private_direction=private_direction,
                participants=participants
            )

            # Build messages array with dynamic content and sprite galleries
            messages = []

            # Add DYNAMIC scene context and private direction FIRST (scene-specific)
            scene_instructions = f"""CURRENT SCENE CONTEXT:
{scene_context}

YOUR PRIVATE DIRECTION:
{private_direction}

Participants in this scene: {', '.join(participants)}"""

            messages.append({
                "role": "user",
                "content": scene_instructions
            })

            # Add conversation history AFTER scene setup
            if conversation_history:
                for exchange in conversation_history[-6:]:  # Last 6 exchanges
                    messages.append({
                        "role": "assistant" if exchange['speaker'] == self.name else "user",
                        "content": f"{exchange['speaker'].upper()}: {exchange['text']}"
                    })

            # Add scene context image (IMG_1402.jpg) if it exists - CACHED
            scene_image_path = Path(__file__).parent.parent / "scene_context" / "IMG_1402.jpg"
            if scene_image_path.exists():
                try:
                    with open(scene_image_path, 'rb') as f:
                        scene_image_data = base64.b64encode(f.read()).decode()

                    scene_context_content = [
                        {
                            "type": "text",
                            "text": "Scene Context Image - This shows the current physical setup:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": scene_image_data
                            },
                            "cache_control": {"type": "ephemeral"}  # Cache the scene context image
                        }
                    ]

                    messages.append({
                        "role": "user",
                        "content": scene_context_content
                    })

                    logger.info("📸 Added scene context image IMG_1402.jpg (CACHED)")
                except Exception as e:
                    logger.error(f"Failed to load scene context image: {e}")

            # Dynamic Content (iOS uploads, web search, etc.) - Never cached
            if dynamic_content_path and Path(dynamic_content_path).exists():
                try:
                    dynamic_content = []

                    with open(dynamic_content_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode()

                    dynamic_content.append({
                        "type": "text",
                        "text": "User uploaded this image for context:"
                    })
                    dynamic_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    })
                    dynamic_content.append({
                        "type": "text",
                        "text": "Consider this in your response."
                        # NO cache_control here - dynamic content should not be cached
                    })

                    messages.append({
                        "role": "user",
                        "content": dynamic_content
                    })

                    logger.info("📱 Added dynamic content from iOS upload (not cached)")
                except Exception as e:
                    logger.error(f"Failed to load dynamic content: {e}")

            # Add sprite galleries and prompt - Sprites CACHED, prompt dynamic
            final_content = []

            # Add sprite galleries first (CACHED)
            if "talking_images.png" in self.sprite_galleries:
                final_content.append({
                    "type": "text",
                    "text": "Here are the sprite galleries for your selection:"
                })
                final_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.sprite_galleries["talking_images.png"]
                    }
                })
                final_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.sprite_galleries["idle_images.png"]
                    },
                    "cache_control": {"type": "ephemeral"}  # Cache up to and including sprite galleries
                })

            # Add the actual prompt (NOT CACHED - dynamic per request)
            final_content.append({
                "type": "text",
                "text": prompt
            })

            messages.append({
                "role": "user",
                "content": final_content
            })

            # Debug log the API call
            logger.info(f"📤 Making API call for Laura with caching")
            logger.info(f"   System blocks: {len(system_blocks)} (cached)")
            logger.info(f"   Messages: {len(messages)} (scene image + sprites cached)")
            logger.info(f"   Cache boundaries: System end, Scene image, Sprite galleries")

            # Make API call with properly consolidated cache
            response = await self.llm_adapter.generate_response(
                messages=messages,
                system_blocks=system_blocks,  # Single cache boundary at end
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
            logger.error(f"❌ CRITICAL ERROR in Laura's generate_response: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Re-raise the error instead of using fallback
            raise e

    def _parse_speaker_tag(self, text: str, participants: List[str]) -> Tuple[str, str]:
        """
        Parse JSON response with sprite selections
        """
        try:
            # Ensure text is not None or empty
            if not text:
                logger.error("Empty response text received")
                raise ValueError("Empty response from API")

            # Strip markdown code blocks if present
            clean_text = text.strip()
            if clean_text.startswith("```"):
                # Remove ```json or ``` from start and ``` from end
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]  # Remove ```json
                elif clean_text.startswith("```"):
                    clean_text = clean_text[3:]  # Remove ```
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]  # Remove trailing ```
                clean_text = clean_text.strip()

            # Try to parse as JSON
            response_data = json.loads(clean_text)

            # Extract fields
            dialogue = response_data.get("dialogue", text)
            next_speaker = response_data.get("next_speaker", "claude").lower()
            speaking_image = response_data.get("speaking_image", "excited01.png")
            idle_image = response_data.get("idle_image", "scheming01.png")

            # LOG SPRITE SELECTIONS
            logger.info("🎨 ===== LAURA'S SPRITE SELECTION =====")
            logger.info(f"📥 Raw JSON response: {json.dumps(response_data, indent=2)}")
            logger.info(f"🗣️ SPEAKING IMAGE SELECTED: {speaking_image}")
            logger.info(f"😴 IDLE IMAGE SELECTED: {idle_image}")
            logger.info("🎨 ======================================")

            # Store images for distribution
            self.last_speaking_image = speaking_image
            self.last_idle_image = idle_image

            # Validate speaker - MUST be in participants list
            if next_speaker not in participants:
                logger.warning(f"Laura selected invalid speaker: {next_speaker} (not in {participants})")
                other_participants = [p for p in participants if p != self.name]
                next_speaker = other_participants[0] if other_participants else "claude"
                logger.info(f"Corrected next_speaker to: {next_speaker}")

            # CRITICAL: Prevent self-selection
            if next_speaker == self.name:
                logger.warning(f"Laura tried to select herself again! Fixing...")
                other_participants = [p for p in participants if p != self.name]
                next_speaker = other_participants[0] if other_participants else "claude"

            return next_speaker, dialogue

        except json.JSONDecodeError as e:
            # NO FALLBACKS - ERROR OUT COMPLETELY
            logger.error(f"❌ CRITICAL JSON DECODE ERROR in Laura's response")
            logger.error(f"Raw response: {repr(text[:500]) if text else 'None'}")
            logger.error(f"Cleaned text: {repr(clean_text[:500]) if 'clean_text' in locals() else 'Not reached'}")
            logger.error(f"JSON error: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            # Re-raise the error to stop the scene
            raise ValueError(f"Laura agent failed to generate valid JSON response: {e}") from e

    def _get_fallback_response(self) -> str:
        """Required by base class - but we don't use fallbacks, we error out"""
        raise NotImplementedError("Laura agent should never use fallback responses - proper errors only!")

    def get_character_definition(self) -> Dict[str, str]:
        """Return character definition from loaded data"""
        # Extract relevant parts from loaded character data
        personality = self.character_data.get("personality", {})

        return {
            "personality": "\n".join(personality.get("core_traits", [])),
            "speaking_style": "\n".join(personality.get("speaking_style", {}).get("patterns", [])),
            "argument_style": "\n".join(personality.get("argument_style", {}).get("tactics", [])),
            "signature_moves": "\n".join(personality.get("argument_style", {}).get("signature_moves", [])),
            "relationships": json.dumps(self.character_data.get("relationships", {}))
        }

    def get_scene_specific_modifiers(self, scene_intensity: str) -> Dict[str, Any]:
        """Adjust Laura's chaos based on scene intensity"""
        modifiers = {
            "low": {
                "chaos_level": "bubbling",
                "tangent_frequency": "occasional",
                "water_urgency": "gentle",
                "signature_phrase": "Just saying..."
            },
            "medium": {
                "chaos_level": "vibrating",
                "tangent_frequency": "frequent",
                "water_urgency": "insistent",
                "signature_phrase": "OK but seriously though—"
            },
            "high": {
                "chaos_level": "full typhoon",
                "tangent_frequency": "constant",
                "water_urgency": "aggressive",
                "signature_phrase": "I WILL DIE ON THIS HILL—"
            }
        }

        return modifiers.get(scene_intensity, modifiers["medium"])

