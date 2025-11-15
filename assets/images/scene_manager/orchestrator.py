#!/usr/bin/env python3
"""
Scene Orchestrator - Director-First Architecture
Director makes all decisions BEFORE agents speak
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
    turn_number: int
    speaker: str
    text: str
    next_speaker: str
    generated_at: Optional[float] = None
    distributed_at: Optional[float] = None
    mood: Optional[str] = None  # For Claude
    speaking_image: Optional[str] = None  # For Laura
    idle_image: Optional[str] = None  # For Laura

class SceneOrchestrator:
    """
    Enhanced orchestrator that uses director guidance for scene management
    Implements proper character context passing with caching
    """

    def __init__(self,
                 mac_hub_bridge=None,
                 director_api_key: str = None,
                 agents: Dict[str, Any] = None):
        """
        Initialize enhanced orchestrator

        Args:
            mac_hub_bridge: Hub bridge for distribution
            director_api_key: API key for director LLM calls
            agents: Dict of character name -> agent instance
        """
        self.mac_hub_bridge = mac_hub_bridge
        self.agents = agents or {}

        # Initialize the REAL director that actually directs
        self.scene_director = DirectorAgent(api_key=director_api_key) if director_api_key else None

        # Scene data from director
        self.scene_data = None
        self.shared_context = None
        self.private_directions = {}

        # Configuration
        self.max_turns = 8  # Default scene length
        self.topic = ""

        # State management
        self.exchanges: Dict[int, DialogueExchange] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.orchestration_active = False
        self.generation_lock = asyncio.Lock()
        self.stop_event = asyncio.Event()

        # PROPER STATE TRACKING
        self.scene_state = {
            "last_completed_turn": 0,
            "last_distributed_turn": 0,
            "last_generated_turn": 0,
            "current_speaker": None,
            "waiting_for_turn": None  # Which turn we're waiting to complete
        }

        logger.info(f"Enhanced orchestrator initialized with agents: {list(self.agents.keys())}")

    async def orchestrate_scene(self,
                               topic: str,
                               participants: List[str] = None,
                               intensity: str = "medium",
                               max_turns: int = 8) -> bool:
        """
        Start scene orchestration with director guidance

        Args:
            topic: Scene topic
            participants: List of participant names
            intensity: Scene intensity (low/medium/high)
            max_turns: Maximum turns in scene

        Returns:
            Success boolean
        """
        try:
            self.orchestration_active = True
            self.topic = topic
            self.max_turns = max_turns
            self.stop_event.clear()

            # Default participants if not specified
            if not participants:
                participants = list(self.agents.keys())

            logger.info("=" * 50)
            logger.info(f"🎬 Starting enhanced scene orchestration")
            logger.info(f"📝 Topic: {topic}")
            logger.info(f"🎭 Participants: {participants}")
            logger.info(f"🔥 Intensity: {intensity}")

            # Step 1: Director analyzes scene and creates context/directions
            if self.scene_director:
                logger.info("🎬 Director analyzing scene...")
                self.scene_data = await self.scene_director.initialize_scene(
                    topic=topic,
                    participants=participants,
                    intensity=intensity
                )

                # Extract JSON data for scene setup
                scene_setup = self.scene_data.get("scene_setup", {})
                participant_dirs = self.scene_data.get("participant_directions", {})
                scene_dynamics = self.scene_data.get("scene_dynamics", {})

                # Build shared context from scene setup
                self.shared_context = f"""Setting: {scene_setup.get('setting', 'unknown')}
Situation: {scene_setup.get('situation', 'unknown')}
Atmosphere: {scene_setup.get('atmosphere', 'tense')}
Emotional Temperature: {scene_setup.get('emotional_temperature', 'medium')}"""

                # Extract private directions for each participant
                for participant in participants:
                    if participant in participant_dirs:
                        p_dir = participant_dirs[participant]
                        self.private_directions[participant] = f"""Your motivation: {p_dir.get('motivation', 'win the argument')}
Your tactic: {p_dir.get('tactic', 'use logic')}
Intensity: {p_dir.get('intensity_modifier', 'normal')}"""
                        if p_dir.get('secret_knowledge'):
                            self.private_directions[participant] += f"\nSecret: {p_dir['secret_knowledge']}"

                # Get first speaker from director's decision
                first_speaker = scene_dynamics.get("first_speaker", "claude")
                logger.info(f"🎬 Director set scene. First speaker: {first_speaker}")

            else:
                # Fallback without director
                logger.info("⚠️ No director available, using basic scene setup")
                self.shared_context = f"Topic: {topic}"
                self.private_directions = {p: "Be yourself" for p in participants}
                # Randomly choose first speaker
                import random
                first_speaker = random.choice(participants)

            # Step 2: Bootstrap with first 2 turns
            logger.info("🥾 Bootstrapping scene with first 2 turns...")

            # Generate turn 1 with first speaker
            await self.generate_and_distribute_turn(turn=1, speaker=first_speaker)

            # Turn 2 will be whoever turn 1 selected
            if 1 in self.exchanges and self.exchanges[1].next_speaker:
                # Create task for turn 2 generation to avoid blocking
                asyncio.create_task(self.generate_and_distribute_turn(
                    turn=2,
                    speaker=self.exchanges[1].next_speaker
                ))
                # Give it a moment to start
                await asyncio.sleep(0.1)

            logger.info("✅ Bootstrap initiated. Scene now event-driven.")

            # Wait for stop signal (scene runs via events)
            await self.stop_event.wait()

            logger.info("🛑 Scene orchestration stopped")
            return True

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            return False
        finally:
            self.orchestration_active = False

    async def generate_and_distribute_turn(self, turn: int, speaker: str = None):
        """
        DIRECTOR-FIRST: Director decides everything, then agent follows instructions

        Args:
            turn: Turn number
            speaker: IGNORED - Director chooses speaker
        """
        async with self.generation_lock:
            logger.info(f"🎬 Turn {turn}: Asking Director for decisions...")

            # STEP 1: DIRECTOR DECIDES EVERYTHING
            if self.scene_director:
                # Get last speaker from history
                last_speaker = None
                if self.conversation_history:
                    last_speaker = self.conversation_history[-1].get("speaker")

                # Director makes all decisions
                decision = await self.scene_director.direct_next_turn(
                    conversation_history=self.conversation_history,
                    last_speaker=last_speaker
                )

                # Extract Director's choices
                speaker = decision.chosen_speaker
                director_instruction = self.scene_director.format_instruction_for_agent(decision)
                logger.info(f"🎯 Director chose speaker: {speaker} for turn {turn}")

                logger.info(f"🎬 Director Decision:")
                logger.info(f"   Speaker: {speaker}")
                logger.info(f"   Instruction: {decision.speaker_instruction}")
                logger.info(f"   Tone: {decision.emotional_direction}")
                if decision.reasoning:
                    logger.info(f"   Reasoning: {decision.reasoning}")
            else:
                # Fallback if no Director
                speaker = "claude" if turn % 2 == 1 else "laura"
                director_instruction = ""
                logger.warning("⚠️ No Director - using fallback alternation")

            # Check agent exists
            if speaker not in self.agents:
                logger.error(f"Agent '{speaker}' not found!")
                return

            agent = self.agents[speaker]

            logger.info(f"🤖 {speaker} generating response with Director's instruction...")
            start_time = time.time()

            try:
                # Build prompt with Director's instruction FIRST
                prompt_lines = []

                # Add Director's instruction prominently
                if director_instruction:
                    prompt_lines.append(director_instruction)
                    prompt_lines.append("")

                # Then conversation history
                if self.conversation_history:
                    prompt_lines.append("Recent conversation:")
                    for entry in self.conversation_history[-6:]:
                        prompt_lines.append(f"{entry['speaker'].upper()}: {entry['text']}")
                    prompt_lines.append("")

                prompt_lines.append(f"Generate your next line of dialogue following the Director's instruction.")
                prompt_lines.append("Output valid JSON with your spoken words only.")
                prompt_lines.append("Keep dialogue brief (1-2 sentences).")

                prompt = "\n".join(prompt_lines)

                # Generate response with Director's instruction embedded in prompt
                # The Director's instruction is already in the prompt, and agents
                # have static character definitions cached in their system blocks
                next_speaker, response_text = await agent.generate_response(
                    prompt=prompt,
                    scene_context=self.shared_context or f"Topic: {self.topic}",
                    private_direction=director_instruction,  # Pass Director's instruction as the direction
                    participants=list(self.agents.keys()),
                    conversation_history=self.conversation_history
                )

                if not response_text:
                    logger.error(f"Failed to generate text for turn {turn}")
                    return

                # Create exchange with agent-specific fields
                exchange = DialogueExchange(
                    turn_number=turn,
                    speaker=speaker,
                    text=response_text,
                    next_speaker=next_speaker,
                    generated_at=time.time()
                )

                # Add agent-specific visual data
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

                # Get per-turn orchestration decision from director
                if self.scene_director and turn > 2:  # Let first 2 turns play naturally
                    turn_decision = await self.scene_director.orchestrate_turn(
                        turn_number=turn,
                        last_speaker=speaker,
                        last_dialogue=response_text,
                        participants=list(self.agents.keys())
                    )

                    # Check for interventions
                    intervention = turn_decision.get("intervention", {})
                    if intervention.get("needed"):
                        logger.info(f"🎬 Director intervention: {intervention.get('type')} - {intervention.get('instruction')}")
                        # Could modify next speaker or add special instructions
                        if intervention.get("enforce_speaker"):
                            next_speaker = intervention["enforce_speaker"]

                gen_time = time.time() - start_time
                # UPDATE GENERATION STATE
                self.scene_state["last_generated_turn"] = max(
                    self.scene_state["last_generated_turn"],
                    turn
                )
                logger.info(f"✅ Generated turn {turn} in {gen_time:.2f}s")
                logger.info(f"   {speaker}: {response_text[:80]}...")
                logger.info(f"   Next speaker: {next_speaker}")
                logger.info(f"📊 Updated last_generated_turn to {self.scene_state['last_generated_turn']}")

                # Distribute immediately
                await self._distribute_turn(exchange)

            except Exception as e:
                logger.error(f"Failed to generate turn {turn}: {e}", exc_info=True)

    async def _distribute_turn(self, exchange: DialogueExchange):
        """
        Distribute turn to hub for device playback

        Args:
            exchange: The dialogue exchange to distribute
        """
        try:
            hub_url = "http://localhost:8766/api/dialogue/distribute_single"

            payload = {
                "speaker": exchange.speaker,
                "turn": exchange.turn_number,
                "text": exchange.text,
                "is_first": exchange.turn_number == 1,
                "next_speaker": exchange.next_speaker  # Include for hub tracking
            }

            # Add agent-specific visual fields
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
                    # UPDATE DISTRIBUTION STATE
                    self.scene_state["last_distributed_turn"] = max(
                        self.scene_state["last_distributed_turn"],
                        exchange.turn_number
                    )
                    logger.info(f"📤 Distributed turn {exchange.turn_number} to {exchange.speaker}")
                    logger.info(f"📊 Updated last_distributed_turn to {self.scene_state['last_distributed_turn']}")
                else:
                    logger.error(f"Failed to distribute turn {exchange.turn_number}: {response.status_code}")

        except Exception as e:
            logger.error(f"Distribution error for turn {exchange.turn_number}: {e}")

    async def handle_playback_complete(self, completed_turn: int):
        """
        Handle playback completion event - triggers N+2 generation

        Args:
            completed_turn: Turn number that just completed
        """
        logger.info(f"🔔 Handling playback complete for turn {completed_turn}")

        # UPDATE REAL STATE
        self.scene_state["last_completed_turn"] = max(self.scene_state["last_completed_turn"], completed_turn)
        logger.info(f"📊 Updated last_completed_turn to {self.scene_state['last_completed_turn']}")

        # Ignore stale events - don't process completions for turns we've already passed
        if completed_turn < self.scene_state["last_completed_turn"] - 1:
            logger.warning(f"⚠️ Ignoring stale playback_complete for turn {completed_turn}")
            return

        # Calculate N+2
        next_turn = completed_turn + 2

        # Check if we should generate
        if next_turn <= self.max_turns and next_turn not in self.exchanges:
            # Determine speaker from previous exchange
            if completed_turn in self.exchanges:
                prev_exchange = self.exchanges[completed_turn]
                # Turn N+1 should be speaking now, so N+2 is whoever they pick
                # But we need to know who N+1 picked...
                if (completed_turn + 1) in self.exchanges:
                    turn_n_plus_1 = self.exchanges[completed_turn + 1]
                    next_speaker = turn_n_plus_1.next_speaker

                    # CRITICAL: Enforce alternation for 2-speaker scenes
                    participants = list(self.agents.keys())
                    if len(participants) == 2:
                        # Force alternation between two speakers
                        current_speaker = turn_n_plus_1.speaker
                        if next_speaker == current_speaker:
                            logger.warning(f"⚠️ Turn {next_turn}: {current_speaker} tried to pick themselves! Forcing alternation.")
                            next_speaker = [p for p in participants if p != current_speaker][0]
                else:
                    # Fallback
                    next_speaker = "laura" if prev_exchange.speaker == "claude" else "claude"
            else:
                # Fallback to rotation
                next_speaker = "claude" if next_turn % 2 == 1 else "laura"

            logger.info(f"📝 Generating turn {next_turn} (N+2 pattern) - Speaker: {next_speaker}")
            # Generate asynchronously to avoid blocking the webhook response
            asyncio.create_task(self.generate_and_distribute_turn(turn=next_turn, speaker=next_speaker))

        elif next_turn > self.max_turns:
            logger.info(f"🏁 Reached max turns ({self.max_turns}), pausing orchestration")
            await self.stop(clear_data=False)  # Preserve data for continuation

    async def stop(self, clear_data: bool = True):
        """Stop the orchestration gracefully

        Args:
            clear_data: Whether to clear scene data (False when pausing for continuation)
        """
        if self.orchestration_active:
            logger.info(f"Stopping orchestration (clear_data={clear_data})...")
            self.orchestration_active = False
            self.stop_event.set()

            if clear_data:
                # Reset agents
                for agent in self.agents.values():
                    agent.reset_scene()

                # Clear scene data
                self.scene_data = None
                self.shared_context = None
                self.private_directions = {}

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            "active": self.orchestration_active,
            "topic": self.topic,
            "turns_generated": len(self.exchanges),
            "max_turns": self.max_turns,
            "participants": list(self.agents.keys()),
            "latest_turn": max(self.exchanges.keys()) if self.exchanges else 0,
            "has_director": self.scene_director is not None,
            "scene_intensity": self.scene_data.get("scene_setup", {}).get("emotional_temperature", "unknown") if self.scene_data else "N/A"
        }