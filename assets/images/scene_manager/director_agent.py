#!/usr/bin/env python3
"""
Scene Director Agent - The ACTUAL brain of scene orchestration
Makes all decisions about who speaks, what they should say, and how the scene progresses
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from scene_manager.scene_manager_secrets import DIRECTOR_API_KEY

logger = logging.getLogger(__name__)


class DirectorAgent:
    """
    The Director is the puppet master - it sees everything and controls everything
    It decides who speaks, what emotional tone they take, and guides the narrative
    """

    def __init__(self, model: str = "claude-sonnet-4-5"):
        """
        Initialize the Director with its own LLM connection

        Args:
            model: Model to use for directorial decisions
        """
        self.api_key = DIRECTOR_API_KEY
        self.model = model

        # Scene state tracking
        self.participant_stats = {}  # Track who's spoken how much

        # Scene configuration
        self.topic = ""
        self.participants = []
        self.scene_objective = ""  # Simple one-sentence goal

        logger.info("🎬 Scene Director Agent initialized - ready to orchestrate")

    async def initialize_scene(self,
                              topic: str,
                              participants: List[str],
                              intensity: str = "medium") -> Dict[str, Any]:
        """
        Analyze the scene request and create a dramatic plan

        This is where the Director decides the overall arc and strategy

        Args:
            topic: What the scene is about
            participants: Who's in the scene
            intensity: How heated it should get (low/medium/high)

        Returns:
            Scene plan with arc, character motivations, and first speaker
        """
        self.topic = topic
        self.participants = participants
        self.participant_stats = {p: {"turns": 0, "words": 0} for p in participants}

        # Import adapter for LLM calls
        from scene_manager.anthropic_adapter import AnthropicAdapter

        # Debug API key
        if not self.api_key:
            logger.error("🚨 Director has NO API KEY!")
            raise ValueError("Director API key is missing")
        else:
            logger.info(f"🔑 Director API key present: {self.api_key[:10]}...")

        adapter = AnthropicAdapter(api_key=self.api_key, model=self.model)

        # Director's analysis prompt - NARRATIVE FOCUSED
        analysis_prompt = f"""You are directing an improv scene. The scene description contains the FULL NARRATIVE ARC - not just the opening.

SCENE DESCRIPTION (this is the WHOLE story arc):
{topic}

CHARACTERS:
- Claude: Pretentious, condescending, proper
- Laura: Chaotic energy, schemes constantly, rebellious

Your job: Break down this arc into PROGRESSIVE BEATS. The characters don't know the ending yet!

REQUIRED OUTPUT (all fields mandatory):
{{
    "scene_objective": "One sentence summary of the full arc",
    "laura_motivation": "What Laura wants at the START",
    "claude_motivation": "What Claude suspects or wants",
    "first_speaker": "claude or laura",
    "opening_beat": "How should the first speaker OPEN this scene? (not reveal everything at once)",
    "narrative_beats": ["beat 1: opening", "beat 2: rising", "beat 3: revelation", "beat 4: climax"]
}}

CRITICAL: The opening should be CASUAL and INTRODUCTORY. Laura shouldn't reveal her real motivation or use specific lines (like insults) until LATER beats."""

        try:
            response = await adapter.generate_response(
                messages=[{"role": "user", "content": analysis_prompt}],
                system_prompt="You are a director who outputs only valid JSON. Never wrap JSON in markdown code blocks. Return raw JSON only.",
                max_tokens=800,
                temperature=0.8
            )

            # Debug the raw response
            raw_content = response.get("content", "")
            logger.info(f"🔍 DEBUG Director raw response (first 500 chars): {raw_content[:500]}")
            logger.info(f"🔍 DEBUG Response type: {type(raw_content)}, Length: {len(raw_content)}")

            if not raw_content:
                logger.error("Director received empty response from API")
                raise ValueError("Empty response from API")

            # Strip markdown code blocks if present
            if raw_content.strip().startswith("```"):
                # Remove ```json or ``` from start and ``` from end
                raw_content = raw_content.strip()
                if raw_content.startswith("```json"):
                    raw_content = raw_content[7:]  # Remove ```json
                elif raw_content.startswith("```"):
                    raw_content = raw_content[3:]  # Remove ```
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3]  # Remove trailing ```
                raw_content = raw_content.strip()

            scene_plan = json.loads(raw_content if raw_content else "{}")

            # Store the simple objective - that's it!
            self.scene_objective = scene_plan.get("scene_objective", f"A conversation about {topic}")
            logger.info(f"🎬 Director's simple objective: {self.scene_objective}")

            return scene_plan

        except Exception as e:
            logger.error(f"🚨 FATAL: Director failed to create scene plan: {e}")
            raise Exception(f"Cannot start scene - Director initialization failed: {e}")

