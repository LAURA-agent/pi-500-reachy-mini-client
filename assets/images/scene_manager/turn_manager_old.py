#!/usr/bin/env python3
"""
Clean Scene Orchestrator - Director-First Architecture
The Director makes ALL decisions, agents just act them out
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
    next_speaker: str  # Who speaks next (needed for bootstrap)
    director_instruction: str  # What the Director told them
    emotional_tone: str       # How the Director said to deliver it
    generated_at: float
    distributed_at: Optional[float] = None
    # Visual data for display
    mood: Optional[str] = None  # Claude's mood
    speaking_image: Optional[str] = None  # Laura's sprite
    idle_image: Optional[str] = None  # Laura's idle sprite

class TurnManager:
    """
    Simple turn manager - runs the scene loop and calls agents
    Does NOT make LLM calls itself - that's what agents do
    """

    def __init__(self,
                 director_api_key: str,
                 agents: Dict[str, Any],
                 max_turns: int = 8):
        """
        Initialize the orchestrator

        Args:
            director_api_key: API key for the Director
            agents: Dict of name -> agent instance
            max_turns: Maximum turns in a scene
        """
        # The Director is the brain
        self.director = DirectorAgent(api_key=director_api_key)

        # Agents are just actors
        self.agents = agents

        # Scene configuration
        self.max_turns = max_turns
        self.topic = ""
        self.participants = []

        # Scene state
        self.exchanges: Dict[int, DialogueExchange] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.scene_active = False
        self.orchestration_active = False  # Compatibility with scene_server
        self.stop_event = asyncio.Event()

        # State tracking for continuation logic
        self.scene_state = {
            "last_completed_turn": 0,
            "last_distributed_turn": 0,
            "last_generated_turn": 0
        }

        logger.info(f"🎬 Clean Orchestrator initialized with agents: {list(agents.keys())}")

    async def run_scene(self,
                        topic: str,
                        participants: Optional[List[str]] = None,
                        intensity: str = "medium") -> bool:
        """
        Run a complete scene with Director control

        Args:
            topic: What the scene is about
            participants: Who's in it (defaults to all agents)
            intensity: How heated (low/medium/high)

        Returns:
            Success boolean
        """
        try:
            self.scene_active = True
            self.orchestration_active = True  # Compatibility
            self.topic = topic
            self.participants = participants or list(self.agents.keys())
            self.stop_event.clear()

            logger.info("=" * 50)
            logger.info("🎬 STARTING SCENE - DIRECTOR-FIRST ARCHITECTURE")
            logger.info(f"📝 Topic: {topic}")
            logger.info(f"🎭 Participants: {self.participants}")
            logger.info(f"🔥 Intensity: {intensity}")
            logger.info("=" * 50)

            # Step 1: Director creates scene plan
            logger.info("🎬 Director creating scene plan...")
            try:
                scene_plan = await self.director.initialize_scene(
                    topic=topic,
                    participants=self.participants,
                    intensity=intensity
                )
                logger.info(f"📋 Scene Plan Created:")
                logger.info(f"   First Speaker: {scene_plan.get('first_speaker', 'unknown')}")
                logger.info(f"   Opening Direction: {scene_plan.get('opening_line_direction', 'unknown')}")

                # STORE the Director's decisions
                self.first_speaker = scene_plan.get('first_speaker')
                self.scene_objective = scene_plan.get('scene_objective', topic)

            except Exception as e:
                logger.error(f"🚨 Scene initialization failed: {e}")
                raise  # Propagate the error - don't continue with broken scene

            # Step 2: Bootstrap with first 2 turns - MUST be sequential for conversation history
            logger.info("🥾 Bootstrapping scene with first 2 turns...")

            # Generate turn 1 first (conversation history is empty)
            turn1_exchange = await self._generate_turn_no_distribute(1)

            # Generate turn 2 second (now sees turn 1 in conversation history)
            turn2_exchange = await self._generate_turn_no_distribute(2)

            # Distribute in order
            await self._distribute_turn(turn1_exchange)
            await self._distribute_turn(turn2_exchange)

            logger.info("✅ Bootstrap complete. Both turns generated and distributed.")

            # Wait for scene to complete via events
            await self.stop_event.wait()

            logger.info("🏁 Scene complete!")
            return True

        except Exception as e:
            logger.error(f"Scene failed: {e}", exc_info=True)
            return False
        finally:
            self.scene_active = False
            self.orchestration_active = False

    async def _generate_turn_no_distribute(self, turn: int) -> DialogueExchange:
        """
        Generate dialogue for a turn but DON'T distribute it
        Used during bootstrap to ensure correct distribution order

        Args:
            turn: Turn number

        Returns:
            DialogueExchange ready for distribution
        """
        logger.info(f"\n--- TURN {turn} ---")

        # Determine speaker
        if turn == 1 and hasattr(self, 'first_speaker') and self.first_speaker:
            speaker = self.first_speaker
            logger.info(f"   Speaker: {speaker} (Director's choice)")
        elif len(self.participants) == 2:
            if self.conversation_history:
                last_speaker = self.conversation_history[-1]["speaker"]
                other_participants = [p for p in self.participants if p != last_speaker]
                speaker = other_participants[0] if other_participants else self.participants[1]
            else:
                speaker = self.participants[0]
            logger.info(f"   Speaker: {speaker} (alternation)")
        else:
            if self.conversation_history:
                last_speaker = self.conversation_history[-1]["speaker"]
                last_idx = self.participants.index(last_speaker)
                speaker = self.participants[(last_idx + 1) % len(self.participants)]
            else:
                speaker = self.first_speaker if hasattr(self, 'first_speaker') and self.first_speaker else self.participants[0]
            logger.info(f"   Speaker: {speaker} (rotation)")

        # Get agent
        if speaker not in self.agents:
            logger.error(f"Speaker {speaker} not found in agents")
            return None

        agent = self.agents[speaker]
        logger.info(f"🎭 {speaker} generating response...")

        # Build prompt
        prompt = self._build_agent_prompt_simple()

        # Generate response
        try:
            next_speaker, response_text = await agent.generate_response(
                prompt=prompt,
                scene_context=f"Topic: {self.topic}\nObjective: {getattr(self.director, 'scene_objective', self.topic)}",
                private_direction=None,
                participants=self.participants,
                conversation_history=self.conversation_history
            )

            logger.info(f"✅ {speaker}: {response_text[:100]}...")

            # Create exchange
            exchange = DialogueExchange(
                turn_number=turn,
                speaker=speaker,
                text=response_text,
                next_speaker=next_speaker,
                director_instruction="autonomous",
                emotional_tone="natural",
                generated_at=time.time()
            )

            # Add visual data
            if speaker == "claude" and hasattr(agent, 'last_mood'):
                exchange.mood = agent.last_mood
            elif speaker == "laura" and hasattr(agent, 'last_speaking_image'):
                exchange.speaking_image = agent.last_speaking_image
                exchange.idle_image = agent.last_idle_image

            # Store exchange and update history
            self.exchanges[turn] = exchange
            self.conversation_history.append({
                "speaker": speaker,
                "text": response_text
            })

            # Update state
            self.scene_state["last_generated_turn"] = turn

            return exchange

        except Exception as e:
            logger.error(f"Agent {speaker} failed to generate: {e}")
            return None

    async def generate_turn(self, turn: int):
        """
        Generate a single turn with simple alternation (Director only at scene start)

        Args:
            turn: Turn number
        """
        logger.info(f"\n--- TURN {turn} ---")

        # For turn 1, use Director's first_speaker decision
        if turn == 1 and hasattr(self, 'first_speaker') and self.first_speaker:
            speaker = self.first_speaker
            logger.info(f"   Speaker: {speaker} (Director's choice)")
        # For turn 2+, use simple alternation based on last speaker
        elif len(self.participants) == 2:
            if self.conversation_history:
                # Alternate from whoever spoke last
                last_speaker = self.conversation_history[-1]["speaker"]
                other_participants = [p for p in self.participants if p != last_speaker]
                speaker = other_participants[0] if other_participants else self.participants[1]
            else:
                # Fallback if no history (shouldn't happen after turn 1)
                speaker = self.participants[0]
            logger.info(f"   Speaker: {speaker} (alternation)")
        else:
            # For multi-speaker scenes, use last speaker's next_speaker hint
            if self.conversation_history:
                last_speaker = self.conversation_history[-1]["speaker"]
                # Find who should speak next based on simple rotation
                last_idx = self.participants.index(last_speaker)
                speaker = self.participants[(last_idx + 1) % len(self.participants)]
            else:
                speaker = self.first_speaker if hasattr(self, 'first_speaker') and self.first_speaker else self.participants[0]
            logger.info(f"   Speaker: {speaker} (rotation)")

        # Step 2: Get the chosen agent
        if speaker not in self.agents:
            logger.error(f"Speaker {speaker} not found in agents")
            return

        agent = self.agents[speaker]

        # Step 3: Agent generates response autonomously (Director already set scene context)
        logger.info(f"🎭 {speaker} generating response...")

        # Build the prompt WITHOUT per-turn Director intervention
        prompt = self._build_agent_prompt_simple()

        # Agent generates based on scene context and conversation flow
        try:
            # Pass the initial scene setup but no per-turn Director instructions
            next_speaker, response_text = await agent.generate_response(
                prompt=prompt,
                scene_context=f"Topic: {self.topic}\nObjective: {getattr(self.director, 'scene_objective', self.topic)}",
                private_direction=None,  # No per-turn direction
                participants=self.participants,
                conversation_history=self.conversation_history
            )

            logger.info(f"✅ {speaker}: {response_text[:100]}...")

            # Step 4: Store the exchange
            exchange = DialogueExchange(
                turn_number=turn,
                speaker=speaker,
                text=response_text,
                next_speaker=next_speaker,  # CRITICAL: Need this for bootstrap!
                director_instruction="autonomous",  # No per-turn direction
                emotional_tone="natural",  # Agent decides tone
                generated_at=time.time()
            )

            # Add visual data if available
            if speaker == "claude" and hasattr(agent, 'last_mood'):
                exchange.mood = agent.last_mood
            elif speaker == "laura" and hasattr(agent, 'last_speaking_image'):
                exchange.speaking_image = agent.last_speaking_image
                exchange.idle_image = agent.last_idle_image

            self.exchanges[turn] = exchange
            self.conversation_history.append({
                "speaker": speaker,
                "text": response_text
            })

            # Step 5: Distribute to devices
            await self._distribute_turn(exchange)

            # Update generation state
            self.scene_state["last_generated_turn"] = turn

        except Exception as e:
            logger.error(f"Agent {speaker} failed to generate: {e}")

    def _build_agent_prompt_simple(self) -> str:
        """
        Build prompt for agent WITHOUT per-turn Director instructions

        Returns:
            Formatted prompt string
        """
        prompt_lines = []

        # Just conversation history for context
        if self.conversation_history:
            prompt_lines.append("Recent conversation:")
            for entry in self.conversation_history[-6:]:
                prompt_lines.append(f"{entry['speaker'].upper()}: {entry['text']}")
            prompt_lines.append("")

        # Simple instruction
        prompt_lines.append("Generate your next line of dialogue.")
        prompt_lines.append("Respond with valid JSON containing your dialogue.")
        prompt_lines.append("Keep it brief (1-2 sentences).")

        return "\n".join(prompt_lines)

    async def _distribute_turn(self, exchange: DialogueExchange):
        """
        Send turn to hub for device playback

        Args:
            exchange: The dialogue to distribute
        """
        try:
            hub_url = "http://localhost:8766/api/dialogue/distribute_single"

            payload = {
                "speaker": exchange.speaker,
                "turn": exchange.turn_number,
                "text": exchange.text,
                "is_first": exchange.turn_number == 1,  # Only turn 1 starts immediately
                "director_instruction": exchange.director_instruction,  # Include for logging
                "emotional_tone": exchange.emotional_tone
            }

            # Add visual data if present
            if exchange.mood:
                payload["mood"] = exchange.mood
            if exchange.speaking_image:
                payload["speaking_image"] = exchange.speaking_image
            if exchange.idle_image:
                payload["idle_image"] = exchange.idle_image

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    hub_url,
                    json=payload,
                    timeout=httpx.Timeout(10.0)
                )

                if response.status_code == 200:
                    exchange.distributed_at = time.time()
                    # Update distribution state
                    self.scene_state["last_distributed_turn"] = exchange.turn_number
                    logger.info(f"📤 Distributed turn {exchange.turn_number}")
                else:
                    logger.error(f"Failed to distribute: {response.status_code}")

        except Exception as e:
            logger.error(f"Distribution error: {e}")

    async def generate_and_distribute_turn(self, turn: int, speaker: str):
        """
        Compatibility method for scene_server continuation logic

        Args:
            turn: Turn number to generate
            speaker: Suggested speaker (Director may override)
        """
        # Just call generate_turn - Director decides speaker anyway
        await self.generate_turn(turn)

    async def handle_playback_complete(self, completed_turn: int):
        """
        Handle playback completion for N+2 pattern

        Args:
            completed_turn: The turn that just finished playing
        """
        # Update completion state
        self.scene_state["last_completed_turn"] = max(
            self.scene_state["last_completed_turn"],
            completed_turn
        )
        logger.info(f"🔔 Turn {completed_turn} playback complete")

        # Standard N+2 pattern: completion of turn N triggers generation of turn N+2
        next_turn = completed_turn + 2

        # Generate N+2 if needed
        if next_turn <= self.max_turns and next_turn not in self.exchanges:
            logger.info(f"📝 Generating turn {next_turn} (N+2 pattern)")
            # Run generation as background task so webhook returns immediately
            asyncio.create_task(self.generate_turn(next_turn))
        elif next_turn > self.max_turns:
            logger.info(f"🏁 Reached max turns ({self.max_turns})")

    async def stop(self):
        """Stop the scene gracefully"""
        logger.info("🛑 Stopping scene...")
        self.scene_active = False
        self.orchestration_active = False
        self.stop_event.set()

        # Reset agents
        for agent in self.agents.values():
            if hasattr(agent, 'reset_scene'):
                agent.reset_scene()

    def get_status(self) -> Dict[str, Any]:
        """Get current scene status"""
        return {
            "active": self.scene_active,
            "topic": self.topic,
            "participants": self.participants,
            "turns_completed": len(self.exchanges),
            "max_turns": self.max_turns,
            "latest_turn": max(self.exchanges.keys()) if self.exchanges else 0
        }

# Example usage for testing
async def test_orchestrator():
    """Test the clean orchestrator"""
    from scene_manager.agents.claude_agent import ClaudeAgent
    from scene_manager.agents.laura_agent import LauraAgent

    # Create agents
    agents = {
        "claude": ClaudeAgent(api_key="your_claude_key"),
        "laura": LauraAgent(api_key="your_laura_key")
    }

    # Create orchestrator
    orchestrator = CleanOrchestrator(
        director_api_key="your_director_key",
        agents=agents,
        max_turns=6
    )

    # Run a scene
    await orchestrator.run_scene(
        topic="Who ate the last cookie",
        intensity="medium"
    )

if __name__ == "__main__":
    # For testing
    asyncio.run(test_orchestrator())