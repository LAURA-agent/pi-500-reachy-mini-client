#!/usr/bin/env python3
"""
Clean Turn Manager - Pure Orchestration
Only manages turn flow, agents handle their own context
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import httpx

from scene_manager.director_agent import DirectorAgent

logger = logging.getLogger(__name__) 

@dataclass
class DialogueExchange:
    """Single turn of dialogue"""
    turn_number: int
    speaker: str
    text: str
    next_speaker: str
    generated_at: float
    distributed_at: Optional[float] = None
    # Visual data for display
    mood: Optional[str] = None
    speaking_image: Optional[str] = None
    idle_image: Optional[str] = None
    tool: Optional[Dict[str, Any]] = None  # Laura's tool data

class TurnManager:
    """
    Clean turn orchestrator - ONLY manages turn flow
    Does not manage context, build prompts, or make decisions
    """

    def __init__(self,
                 director_api_key: str,
                 agent_service_urls: Dict[str, str], # NEW: Dictionary of agent_name -> service_url
                 max_turns: int = 8):
        """
        Initialize turn manager

        Args:
            director_api_key: API key for Director
            agent_service_urls: Dict of agent_name -> agent_service_url
            max_turns: Maximum turns in scene
        """
        # Director sets up scenes (gets API key from scene_manager_secrets)
        self.director = DirectorAgent()

        # Agent services for LLM inference
        self.agent_service_urls = agent_service_urls

        # Scene configuration
        self.max_turns = max_turns
        self.topic = ""
        self.participants = list(agent_service_urls.keys()) # Participants are now derived from agent_service_urls

        # Scene state
        self.exchanges: Dict[int, DialogueExchange] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.scene_active = False
        self.stop_event = asyncio.Event()

        # Track what we're waiting for
        self.last_completed_turn = 0
        self.last_distributed_turn = 0

        # Director's scene decisions
        self.first_speaker = None
        self.scene_objective = None
        self.scene_plan = None  # Store full plan

        logger.info(f"🎬 Clean Turn Manager initialized with agent services: {list(agent_service_urls.keys())}")

    async def run_scene(
                       self,
                       topic: str,
                       participants: Optional[List[str]] = None, # This will be ignored, participants from agent_service_urls
                       intensity: str = "medium") -> bool:
        """
        Run a scene with clean orchestration

        Args:
            topic: What to argue about
            participants: Who's in it (will be derived from agent_service_urls)
            intensity: How heated

        Returns:
            Success boolean
        """
        try:
            self.scene_active = True
            self.topic = topic
            # Ensure participants are consistent with agent_service_urls
            if participants and set(participants) != set(self.agent_service_urls.keys()):
                logger.warning(f"Participants {participants} provided to run_scene do not match initialized agents {list(self.agent_service_urls.keys())}. Using initialized agents.")
            self.participants = list(self.agent_service_urls.keys())
            self.stop_event.clear()

            logger.info("="*50)
            logger.info("🎬 STARTING CLEAN SCENE")
            logger.info(f"📝 Topic: {topic}")
            logger.info(f"🎭 Participants: {self.participants}")
            logger.info(f"🔥 Intensity: {intensity}")
            logger.info("="*50)

            # Step 1: Director sets up scene
            logger.info("🎬 Director planning scene...")
            scene_plan = await self.director.initialize_scene(
                topic=topic,
                participants=self.participants,
                intensity=intensity
            )

            # Store Director's decisions
            self.scene_plan = scene_plan
            self.first_speaker = scene_plan.get('first_speaker', self.participants[0])
            self.scene_objective = scene_plan.get('scene_objective', topic)

            logger.info(f"📋 Scene Plan:")
            logger.info(f"   First Speaker: {self.first_speaker}")
            logger.info(f"   Objective: {self.scene_objective}")
            logger.info(f"   Full Plan: {scene_plan}")

            # Step 2: Start generation cascade with Turn 1
            # Turn 1 generation will automatically trigger Turn 2, which triggers Turn 3, etc.
            logger.info("🎬 Starting generation cascade with Turn 1...")

            # Generate Turn 1 (this will cascade to all subsequent turns)
            turn1_exchange = await self._generate_turn(1)

            # Distribute ONLY Turn 1 with is_first=True (plays immediately)
            # Turn 2+ will be distributed by N+2 buffering after playback_complete signals
            await self._distribute_turn(turn1_exchange, is_first=True)

            logger.info("✅ Scene started - Turn 1 distributed, cascade running, N+2 buffering active")

            # Wait for scene to complete
            await self.stop_event.wait()

            logger.info("🏁 Scene complete!")
            return True

        except Exception as e:
            logger.error(f"Scene failed: {e}", exc_info=True)
            return False
        finally:
            self.scene_active = False

    async def _generate_turn(self, turn: int) -> DialogueExchange:
        """
        Generate a single turn of dialogue by requesting text from the agent service.
        CRITICAL: Immediately triggers generation of Turn N+1 when Turn N text completes.

        Args:
            turn: Turn number

        Returns:
            DialogueExchange ready for distribution
        """
        # Determine speaker
        if turn == 1:
            # Director's choice for first speaker
            speaker = self.first_speaker
        else:
            # Let agents choose via next_speaker from previous exchange
            if (turn - 1) in self.exchanges and self.exchanges[turn - 1].next_speaker:
                speaker = self.exchanges[turn - 1].next_speaker
            else:
                # Fallback to alternation if next_speaker not provided or previous exchange missing
                if self.conversation_history:
                    last_speaker = self.conversation_history[-1]["speaker"]
                    others = [p for p in self.participants if p != last_speaker]
                    speaker = others[0] if others else self.participants[0]
                else:
                    speaker = self.participants[0]

        logger.info(f"\n--- TURN {turn}: {speaker} ---")

        # Get agent service URL
        agent_url = self.agent_service_urls.get(speaker)
        if not agent_url:
            logger.error(f"Agent service URL for {speaker} not found")
            raise ValueError(f"Agent service URL for {speaker} not found")

        # Agent generates response via HTTP request
        try:
            # Build scene context with full director guidance
            scene_context_lines = [f"Topic: {self.topic}", f"Objective: {self.scene_objective}"]

            # Add turn-specific context for first turn
            if turn == 1:
                scene_context_lines.append("\nIMPORTANT: This is the OPENING of the scene. Start from the beginning, don't jump to the ending.")
                if self.scene_plan and "opening_beat" in self.scene_plan:
                    scene_context_lines.append(f"Opening direction: {self.scene_plan['opening_beat']}")

            # Add character-specific motivation if available
            if self.scene_plan:
                if speaker == "laura" and "laura_motivation" in self.scene_plan:
                    scene_context_lines.append(f"\nYour motivation: {self.scene_plan['laura_motivation']}")
                elif speaker == "claude" and "claude_motivation" in self.scene_plan:
                    scene_context_lines.append(f"\nYour motivation: {self.scene_plan['claude_motivation']}")

                # Add narrative beats for context
                if "narrative_beats" in self.scene_plan:
                    scene_context_lines.append(f"\nNarrative progression: {', '.join(self.scene_plan['narrative_beats'])}")

            # Build prompt and context for the agent service
            prompt_payload = {
                "prompt": self._build_prompt(),
                "scene_context": "\n".join(scene_context_lines),
                "private_direction": None,  # No per-turn direction
                "participants": self.participants,
                "conversation_history": self.conversation_history,
                "turn": turn  # For tool execution timing
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent_url}/generate_response", # Assuming /generate_response endpoint
                    json=prompt_payload,
                    timeout=httpx.Timeout(30.0) # Increased timeout for LLM inference
                )
                response.raise_for_status() # Raise an exception for bad status codes

                agent_response = response.json()
                next_speaker = agent_response.get("next_speaker")
                response_text = agent_response.get("response_text")
                mood = agent_response.get("mood")
                speaking_image = agent_response.get("speaking_image")
                idle_image = agent_response.get("idle_image")
                tool_data = agent_response.get("tool")  # Laura's tool use

            logger.info(f"✅ Generated: {response_text}")
            if tool_data:
                logger.info(f"🔧 Tool: {tool_data.get('action')} ({tool_data.get('timing')})")

            # Create exchange
            exchange = DialogueExchange(
                turn_number=turn,
                speaker=speaker,
                text=response_text,
                next_speaker=next_speaker,
                generated_at=time.time(),
                mood=mood,
                speaking_image=speaking_image,
                idle_image=idle_image,
                tool=tool_data
            )

            # Store exchange and update history
            self.exchanges[turn] = exchange
            self.conversation_history.append({
                "speaker": speaker,
                "text": response_text
            })

            # CASCADE: Turn N text completes → immediately trigger Turn N+1 generation
            next_turn = turn + 1
            if next_turn <= self.max_turns and next_turn not in self.exchanges:
                logger.info(f"🔄 Text cascade: Turn {turn} complete → triggering Turn {next_turn} generation")
                asyncio.create_task(self._generate_turn(next_turn))

            return exchange

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error generating turn {turn} from {speaker}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate turn {turn} from {speaker}: {e}", exc_info=True)
            raise

    def _build_prompt(self) -> str:
        """
        Build minimal prompt for agent
        Agents handle their own character context
        """
        lines = []

        # Recent conversation for continuity
        if self.conversation_history:
            lines.append("Recent conversation:")
            for entry in self.conversation_history[-6:]:
                lines.append(f"{entry['speaker'].upper()}: {entry['text']}")
            lines.append("")

        # Simple instruction
        lines.append("Generate your next line of dialogue.")
        lines.append("Keep it brief and in character.")

        return "\n".join(lines)

    async def _distribute_turn(self, exchange: DialogueExchange, is_first: bool = False):
        """
        Send turn to devices for playback

        Args:
            exchange: Dialogue to distribute
            is_first: Whether this starts playback immediately
        """
        try:
            import os
            hub_port = int(os.getenv("HUB_PORT", 9001))
            hub_url = f"http://localhost:{hub_port}/api/dialogue/distribute_single"

            payload = {
                "speaker": exchange.speaker,
                "turn": exchange.turn_number,
                "text": exchange.text,
                "is_first": is_first,  # Only turn 1 gets True
            }

            # Add visual data if present
            if exchange.mood:
                payload["mood"] = exchange.mood
            if exchange.speaking_image:
                payload["speaking_image"] = exchange.speaking_image
            if exchange.idle_image:
                payload["idle_image"] = exchange.idle_image
            if exchange.tool:
                payload["tool"] = exchange.tool

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    hub_url,
                    json=payload,
                    timeout=httpx.Timeout(10.0)
                )

                if response.status_code == 200:
                    exchange.distributed_at = time.time()
                    self.last_distributed_turn = exchange.turn_number
                    logger.info(f"📤 Distributed turn {exchange.turn_number} (is_first={is_first})")
                else:
                    logger.error(f"Failed to distribute: {response.status_code}")

        except Exception as e:
            logger.error(f"Distribution error: {e}")

    async def handle_playback_complete(self, completed_turn: int):
        """
        Handle playback completion - ONLY distributes N+2 turn (generation already cascaded).

        playback_complete(N) → distribute Turn N+2 (which should already be generated by cascade)

        Args:
            completed_turn: Turn that just finished playing
        """
        self.last_completed_turn = max(self.last_completed_turn, completed_turn)
        logger.info(f"🔔 Turn {completed_turn} playback complete")

        # Distribute the next turn (should already be generated by cascade)
        next_turn_to_distribute = completed_turn + 1
        if next_turn_to_distribute <= self.max_turns:
            if next_turn_to_distribute in self.exchanges:
                exchange_to_distribute = self.exchanges[next_turn_to_distribute]
                if not exchange_to_distribute.distributed_at:
                    logger.info(f"📤 Distributing pre-generated turn {next_turn_to_distribute}")
                    await self._distribute_turn(exchange_to_distribute, is_first=False)
                else:
                    logger.info(f"⚠️ Turn {next_turn_to_distribute} already distributed")
            else:
                # Cascade should have generated this already - wait for it
                logger.warning(f"⏳ Turn {next_turn_to_distribute} not ready yet, waiting for cascade...")
                for _ in range(100):  # Wait up to 10 seconds
                    if next_turn_to_distribute in self.exchanges:
                        await self._distribute_turn(self.exchanges[next_turn_to_distribute], is_first=False)
                        logger.info(f"📤 Distributed turn {next_turn_to_distribute} after wait")
                        break
                    await asyncio.sleep(0.1)
                else:
                    logger.error(f"❌ Turn {next_turn_to_distribute} never generated by cascade")
        elif completed_turn == self.max_turns:
            logger.info(f"🏁 Final turn complete, ending scene")
            self.stop_event.set()


    async def stop(self):
        """Stop the scene gracefully"""
        logger.info("🛑 Stopping scene...")
        self.scene_active = False
        self.stop_event.set()

        # No need to reset agents as they are external services

    def get_status(self) -> Dict[str, Any]:
        """Get current scene status"""
        return {
            "active": self.scene_active,
            "topic": self.topic,
            "participants": self.participants,
            "turns_completed": self.last_completed_turn,
            "turns_distributed": self.last_distributed_turn,
            "max_turns": self.max_turns
        }