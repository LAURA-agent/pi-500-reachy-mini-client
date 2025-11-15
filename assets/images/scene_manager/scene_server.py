#!/usr/bin/env python3
"""
Scene Server - Director-First Architecture
Coordinates scene orchestration with proper character loading and caching
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Import the enhanced orchestrator and agents
import sys
sys.path.append('/Users/lauras/Desktop/laura')
from scene_manager.turn_manager import TurnManager
from scene_manager.agents.claude_agent import ClaudeAgent
from scene_manager.agents.laura_agent import LauraAgent

# Configure logging - simple format for readability
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Just the message, no metadata noise
)
logger = logging.getLogger(__name__)

# Import secrets for API keys
try:
    from .scene_manager_secrets import CLAUDE_API_KEY, LAURA_API_KEY, DIRECTOR_API_KEY
    logger.info(f"✅ Loaded API keys from scene_manager_secrets.py")
except ImportError:
    try:
        # Try direct import if running as script
        import scene_manager_secrets
        CLAUDE_API_KEY = scene_manager_secrets.CLAUDE_API_KEY
        LAURA_API_KEY = scene_manager_secrets.LAURA_API_KEY
        DIRECTOR_API_KEY = scene_manager_secrets.DIRECTOR_API_KEY
        logger.info(f"✅ Loaded API keys from scene_manager_secrets.py")
    except ImportError:
        # Fallback if secrets file doesn't exist
        CLAUDE_API_KEY = None
        LAURA_API_KEY = None
        DIRECTOR_API_KEY = None
        logger.warning("scene_manager_secrets.py not found - API keys must be provided in requests")

class SceneState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    STOPPING = "stopping"

class SceneManager:
    """Main scene management server"""

    def __init__(self):
        self.orchestrator: Optional[TurnManager] = None
        self.state = SceneState.IDLE
        self.current_scene_id: Optional[str] = None
        self.loaded_agents = {}

    async def start_scene(self, scene_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new scene with director guidance and character context

        Args:
            scene_params: Scene configuration from iOS

        Returns:
            Status dict
        """
        if self.state != SceneState.IDLE:
            raise HTTPException(status_code=400, detail="A scene is already active")

        logger.info(f"🎭 Starting enhanced scene with topic: {scene_params.get('topic')}")

        try:
            # Get API keys from params or use defaults from secrets
            claude_api_key = scene_params.get("claude_api_key") or CLAUDE_API_KEY
            laura_api_key = scene_params.get("laura_api_key") or LAURA_API_KEY
            director_api_key = scene_params.get("director_api_key") or DIRECTOR_API_KEY

            # Validate minimum requirements
            if not claude_api_key or not laura_api_key:
                logger.error("Missing required API keys for agents")
                raise ValueError("Claude and Laura API keys are required")

            if not director_api_key:
                logger.error("🚨 Missing DIRECTOR_API_KEY - scene cannot start without Director")
                raise ValueError("Director API key is required for scene orchestration")

            # Get scene parameters
            topic = scene_params.get("topic", "a friendly debate")
            participants = scene_params.get("participants") or ["claude", "laura"]  # Handle None case
            intensity = scene_params.get("intensity", "medium")
            # Support both num_turns (iOS app) and max_turns (web demo)
            max_turns = scene_params.get("num_turns") or scene_params.get("max_turns", 6)

            # Connect to already-running agent services (BattleBots Demo architecture)
            import os
            claude_port = int(os.getenv("CLAUDE_AGENT_PORT", 9002))
            laura_port = int(os.getenv("LAURA_AGENT_PORT", 9003))

            agent_service_urls = {}
            if "claude" in participants:
                agent_service_urls["claude"] = f"http://localhost:{claude_port}"
            if "laura" in participants:
                agent_service_urls["laura"] = f"http://localhost:{laura_port}"

            logger.info(f"Connecting to agent services: {agent_service_urls}")

            # Create turn manager with Director-first architecture
            self.orchestrator = TurnManager(
                director_api_key=director_api_key,  # Director is the brain
                agent_service_urls=agent_service_urls,  # URLs to agent services
                max_turns=max_turns
            )

            self.state = SceneState.ACTIVE
            self.current_scene_id = f"scene_{int(asyncio.get_event_loop().time())}"

            # Start orchestration with director guidance
            asyncio.create_task(
                self.orchestrator.run_scene(
                    topic=topic,
                    participants=participants,
                    intensity=intensity
                )
            )

            logger.info(f"✅ Scene started: {self.current_scene_id}")

            return {
                "status": "success",
                "scene_id": self.current_scene_id,
                "message": f"Scene started with {len(participants)} participants",
                "topic": topic,
                "intensity": intensity,
                "participants": participants,
                "has_director": director_api_key is not None
            }

        except Exception as e:
            logger.error(f"Failed to start scene: {e}", exc_info=True)
            self.state = SceneState.IDLE
            raise HTTPException(status_code=500, detail=str(e))

    async def stop_scene(self) -> Dict[str, Any]:
        """Stop the current scene"""
        if self.state != SceneState.ACTIVE:
            return {"status": "no_scene", "message": "No active scene to stop"}

        logger.info(f"🛑 Stopping scene: {self.current_scene_id}")
        self.state = SceneState.STOPPING

        try:
            if self.orchestrator:
                await self.orchestrator.stop()

            # Agents are external services, just clear references
            self.loaded_agents = {}

            self.state = SceneState.IDLE
            stopped_id = self.current_scene_id
            self.current_scene_id = None

            return {
                "status": "stopped",
                "scene_id": stopped_id,
                "message": "Scene stopped successfully"
            }

        except Exception as e:
            logger.error(f"Error stopping scene: {e}")
            self.state = SceneState.IDLE
            return {"status": "error", "message": str(e)}

    async def continue_scene(self, additional_turns: int = 6) -> Dict[str, Any]:
        """
        Continue a stopped scene with additional turns

        Args:
            additional_turns: Number of additional turns to add

        Returns:
            Status dict
        """
        try:
            if self.state == SceneState.IDLE:
                # Check if we have a recently stopped scene we can continue
                if not self.orchestrator or not self.loaded_agents:
                    return {
                        "status": "error",
                        "message": "No scene to continue. Please start a new scene."
                    }

                logger.info(f"🔄 Continuing stopped scene with {additional_turns} more turns")

                # Reactivate the scene
                self.state = SceneState.ACTIVE

                # Increase max turns in the orchestrator
                if hasattr(self.orchestrator, 'max_turns'):
                    old_max = self.orchestrator.max_turns
                    self.orchestrator.max_turns = old_max + additional_turns
                    logger.info(f"📈 Increased max turns from {old_max} to {self.orchestrator.max_turns}")

                # Reactivate the orchestrator if it was stopped
                if hasattr(self.orchestrator, 'orchestration_active'):
                    if not self.orchestrator.orchestration_active:
                        logger.info("🔄 Reactivating stopped orchestrator")
                        self.orchestrator.orchestration_active = True
                        self.orchestrator.stop_event.clear()

                # USE REAL STATE instead of guessing
                scene_state = self.orchestrator.scene_state
                last_completed = scene_state["last_completed_turn"]
                last_distributed = scene_state["last_distributed_turn"]
                last_generated = scene_state["last_generated_turn"]

                logger.info(f"📊 Scene state: completed={last_completed}, distributed={last_distributed}, generated={last_generated}")

                # Determine what we need to generate/distribute
                next_turn = last_completed + 1
                participants = list(self.orchestrator.agent_service_urls.keys())

                # Find next speaker based on the last completed turn
                if last_completed in self.orchestrator.exchanges:
                    last_exchange = self.orchestrator.exchanges[last_completed]
                    if len(participants) == 2:
                        # For 2-speaker, who plays next turn depends on parity
                        next_speaker = "claude" if next_turn % 2 == 1 else "laura"
                    else:
                        next_speaker = last_exchange.next_speaker
                else:
                    next_speaker = "claude" if next_turn % 2 == 1 else "laura"

                logger.info(f"🎬 Continuing from completed turn {last_completed}, next is turn {next_turn} by {next_speaker}")

                # Only generate if not already generated
                if next_turn > last_generated:
                    asyncio.create_task(
                        self.orchestrator.generate_and_distribute_turn(
                            turn=next_turn,
                            speaker=next_speaker
                        )
                    )

                    # Also prepare N+2
                    other_speaker = [p for p in participants if p != next_speaker][0]
                    asyncio.create_task(
                        self.orchestrator.generate_and_distribute_turn(
                            turn=next_turn + 1,
                            speaker=other_speaker
                        )
                    )

                return {
                    "status": "continued",
                    "message": f"Scene continued with {additional_turns} additional turns",
                    "scene_state": {
                        "last_completed_turn": last_completed,
                        "last_distributed_turn": last_distributed,
                        "last_generated_turn": last_generated,
                        "next_turn": next_turn,
                        "next_speaker": next_speaker
                    },
                    "new_max_turns": self.orchestrator.max_turns
                }

            elif self.state == SceneState.ACTIVE:
                # Scene is already active, extend max turns and trigger next generation
                if hasattr(self.orchestrator, 'max_turns'):
                    old_max = self.orchestrator.max_turns
                    self.orchestrator.max_turns = old_max + additional_turns
                    logger.info(f"📈 Extended active scene from {old_max} to {self.orchestrator.max_turns} turns")

                    # Find the last completed turn and trigger next generation
                    last_turn = len(self.orchestrator.exchanges)
                    if last_turn >= old_max:
                        # We were stopped at max turns, restart orchestration
                        logger.info(f"🎬 Scene was stopped at turn {last_turn}, restarting orchestration")

                        # Reactivate the orchestrator if it was stopped
                        if hasattr(self.orchestrator, 'orchestration_active'):
                            if not self.orchestrator.orchestration_active:
                                logger.info("🔄 Reactivating stopped orchestrator")
                                self.orchestrator.orchestration_active = True
                                self.orchestrator.stop_event.clear()

                        # Directly generate the next turn
                        participants = list(self.orchestrator.agent_service_urls.keys())
                        if last_turn in self.orchestrator.exchanges:
                            last_exchange = self.orchestrator.exchanges[last_turn]
                            # Simple alternation for 2-speaker scenes
                            if len(participants) == 2:
                                next_speaker = [p for p in participants if p != last_exchange.speaker][0]
                            else:
                                next_speaker = last_exchange.next_speaker
                        else:
                            next_speaker = "claude" if last_turn % 2 == 0 else "laura"

                        logger.info(f"🎬 Extending scene, generating turns {last_turn + 1} and {last_turn + 2}")
                        # Generate BOTH N+1 and N+2 to maintain the pattern
                        asyncio.create_task(
                            self.orchestrator.generate_and_distribute_turn(
                                turn=last_turn + 1,
                                speaker=next_speaker
                            )
                        )

                        # Also generate N+2 immediately
                        other_speaker = [p for p in participants if p != next_speaker][0]
                        asyncio.create_task(
                            self.orchestrator.generate_and_distribute_turn(
                                turn=last_turn + 2,
                                speaker=other_speaker
                            )
                        )

                return {
                    "status": "extended",
                    "message": f"Active scene extended by {additional_turns} turns",
                    "new_max_turns": self.orchestrator.max_turns,
                    "continuing_from": last_turn if 'last_turn' in locals() else None
                }
            else:
                return {
                    "status": "error",
                    "message": f"Cannot continue scene in state: {self.state.value}"
                }

        except Exception as e:
            logger.error(f"Error continuing scene: {e}")
            return {"status": "error", "message": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current scene status"""
        base_status = {
            "state": self.state.value,
            "scene_id": self.current_scene_id,
            "loaded_agents": list(self.loaded_agents.keys())
        }

        if self.state == SceneState.ACTIVE and self.orchestrator:
            orchestrator_status = self.orchestrator.get_status()
            return {**base_status, **orchestrator_status}

        return base_status

# Global scene manager instance
scene_manager = SceneManager()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🎭 Scene Server starting...")
    yield
    logger.info("🎭 Scene Server shutting down...")
    if scene_manager.state != SceneState.IDLE:
        await scene_manager.stop_scene()

app = FastAPI(title="Scene Server", version="3.0.0", lifespan=lifespan)

# Request models
class StartSceneRequest(BaseModel):
    topic: str = "general conversation"
    participants: Optional[List[str]] = None
    intensity: str = "medium"  # low/medium/high
    max_turns: Optional[int] = None  # Web demo uses this
    num_turns: Optional[int] = None  # iOS app uses this
    claude_api_key: Optional[str] = None
    laura_api_key: Optional[str] = None
    director_api_key: Optional[str] = None

# API Endpoints
@app.post("/scene/start")
async def start_scene_endpoint(request: StartSceneRequest):
    """Start a new enhanced scene with director and character context"""
    return await scene_manager.start_scene(request.model_dump())

@app.post("/scene/stop")
async def stop_scene_endpoint():
    """Stop the current scene"""
    return await scene_manager.stop_scene()

@app.post("/scene/continue")
async def continue_scene_endpoint(request: dict):
    """Continue a stopped scene with additional turns"""
    additional_turns = request.get("additional_turns", 6)
    return await scene_manager.continue_scene(additional_turns)

@app.get("/scene/status")
async def get_status_endpoint():
    """Get current scene status"""
    return scene_manager.get_status()

# Webhook for playback events
@app.post("/playback_complete")
async def playback_complete_webhook(request: dict):
    """
    Webhook for playback completion events from gameboy devices
    Drives the N+2 generation pattern
    """
    try:
        completed_turn = request.get("turn")
        from_device = request.get("from_device")

        logger.info(f"📨 Playback complete: turn {completed_turn} from {from_device}")

        if scene_manager.state == SceneState.ACTIVE and completed_turn is not None:
            if scene_manager.orchestrator:
                # Forward to orchestrator for N+2 generation (NON-BLOCKING)
                asyncio.create_task(scene_manager.orchestrator.handle_playback_complete(completed_turn))
                return {"status": "acknowledged", "turn": completed_turn}

        logger.debug(f"Ignoring playback_complete: state={scene_manager.state.value}, turn={completed_turn}")
        return {"status": "ignored", "reason": "No active scene or invalid turn"}

    except Exception as e:
        logger.error(f"Error handling playback_complete: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/playback_started")
async def playback_started_webhook(request: dict):
    """Optional webhook for tracking when playback starts"""
    try:
        turn = request.get("turn")
        from_device = request.get("from_device")
        logger.info(f"🎵 Playback started: turn {turn} from {from_device}")
        return {"status": "acknowledged"}
    except Exception as e:
        logger.error(f"Error handling playback_started: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Scene Server v3.0",
        "mode": "director-guided with character caching",
        "scene_active": scene_manager.state == SceneState.ACTIVE,
        "features": {
            "director": DIRECTOR_API_KEY is not None,
            "caching": True,
            "character_context": True
        }
    }

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8787))

    print("🎭 Starting Scene Server (Director-First Architecture)")
    print(f"📍 API will be available at http://localhost:{port}")
    print("🎬 Director guidance: " + ("Enabled" if DIRECTOR_API_KEY else "Disabled (using fallback)"))
    print("💾 Character caching: Enabled")
    print("\nReady for scene orchestration...")

    uvicorn.run(app, host="0.0.0.0", port=port)