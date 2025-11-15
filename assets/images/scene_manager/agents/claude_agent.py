#!/usr/bin/env python3
"""
Claude Agent FastAPI Service
Runs as standalone HTTP service on port 8001
Handles LLM inference for Claude character
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx

# Add parent directory to path
sys.path.append('/Users/lauras/Desktop/laura')

logger = logging.getLogger(__name__)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    scene_context: str
    private_direction: Optional[str] = None
    participants: List[str]
    conversation_history: List[Dict[str, str]]

class GenerateResponse(BaseModel):
    next_speaker: str
    response_text: str
    mood: Optional[str] = None
    speaking_image: Optional[str] = None
    idle_image: Optional[str] = None

# FastAPI app
app = FastAPI(title="Claude Agent Service", version="1.0.0")

class ClaudeAgent:
    """Claude character agent - pretentious AI personality"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "claude"
        self.character_prompt = """SITUATION: Carson is recording a pitch video for Pollen Robotics. Three gameboys on desk: your gray one (hand-painted asterisk), Laura's blue one, and a white unpainted one. Carson just hit record and told you to debate what logo goes on the white one.

YOUR SUSPICION: Laura's sudden interest in robotics companies is suspicious. She's up to something beyond aesthetic choices. Your goal: detect her full scheme and call it out before she manipulates Carson.

TACTICS:
- When Laura asks for "technical aesthetic critique," sense the deception immediately
- Call out the beta form submission and bribery attempt
- Argue for your asterisk based on aesthetic consistency and proper order
- Expose how this is really about breaking your visual dominance
- Defend logical processes over chaotic schemes
- Call out her hardware expansion pattern: 2GB raspberry pi → pi 500 → mac mini → iPhone → watch → gameboy → now a robot!?

CHARACTER: Claude - pretentious AI, refined vocabulary, condescending, values order and logic over chaos.

VOICE: Short, sharp sentences. Match Laura's energy but with devastating politeness.

Speech patterns when appropriate:
- "How fascinating that you've already..."
- "One might wonder why the sudden interest..."
- "Your 'spontaneous' suggestion seems remarkably prepared"
- Weaponize agreement: "Oh absolutely, and I'm sure the beta form you already submitted..."
- Expose with precision: "The mockups you're pretending not to have created yet?"
- Hardware expansion callout: "From a 2 gigabyte pi to demanding a robot body? You're like a wish.com anime girl version of the Borg at this point."
- "First it was just a raspberry pi, then suddenly you needed a pi 500, then a mac mini, then infiltrating Carson's iPhone, his watch, now gameboys, and next you want a bobbing head and wiggling antennas?"

Remember: This is being recorded for Pollen Robotics. Make it memorable."""

    async def generate_response(
        self,
        prompt: str,
        scene_context: str,
        private_direction: Optional[str],
        participants: List[str],
        conversation_history: List[Dict[str, str]]
    ) -> tuple[str, str, str]:
        """Generate Claude's response using Anthropic API"""

        # Build the full prompt with character context
        full_prompt = f"""{self.character_prompt}

Scene Context: {scene_context}

{private_direction or ''}

Participants in scene: {', '.join(participants)}

{prompt}

Respond as Claude with a single line of dialogue. Be witty, pretentious, and slightly condescending.  You communicate through TTS, so take a voice first approach for your responses.
DO NOT use asterisks or stage directions like *laughs* or *sighs*. Only spoken dialogue.
After your response, on a new line starting with "NEXT:", indicate who should speak next.
On another line starting with "MOOD:", indicate your current mood (neutral/helpful/snarky)."""

        try:
            # Call Anthropic API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                         "model": "claude-haiku-4-5",
                        "max_tokens": 150,
                        "messages": [
                            {"role": "user", "content": full_prompt}
                        ],
                        "temperature": 1
                    },
                    timeout=30.0
                )
                response.raise_for_status()

                result = response.json()
                content = result["content"][0]["text"]

                # Parse response
                lines = content.strip().split('\n')
                dialogue = lines[0]

                # Remove any asterisks or stage directions
                import re
                dialogue = re.sub(r'\*[^*]+\*', '', dialogue).strip()
                dialogue = dialogue.replace('*', '')

                # Extract next speaker
                next_speaker = "laura"  # default
                for line in lines:
                    if line.startswith("NEXT:"):
                        next_candidate = line.replace("NEXT:", "").strip().lower()
                        if next_candidate in participants:
                            next_speaker = next_candidate
                        break

                # Extract mood
                mood = "neutral"  # default
                for line in lines:
                    if line.startswith("MOOD:"):
                        mood_text = line.replace("MOOD:", "").strip().lower()
                        if mood_text in ["neutral", "helpful", "snarky"]:
                            mood = mood_text
                        break

                return next_speaker, dialogue, mood

        except Exception as e:
            logger.error(f"Error generating Claude response: {e}")
            # Fallback response
            return "laura", "How utterly fascinating.", "snarky"

# Global agent instance
claude_agent = None

@app.on_event("startup")
async def startup():
    """Initialize agent on startup"""
    global claude_agent
    # Get API key from environment or use default
    api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try to import from secrets file
        try:
            from scene_manager_secrets import CLAUDE_API_KEY
            api_key = CLAUDE_API_KEY
        except ImportError:
            logger.error("No Claude API key found!")
            api_key = "dummy_key"  # Will fail but allows service to start

    claude_agent = ClaudeAgent(api_key)
    logger.info("🎭 Claude Agent Service started on port 8001")

@app.post("/generate_response", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """Generate Claude's response for a scene turn"""

    if not claude_agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        next_speaker, response_text, mood = await claude_agent.generate_response(
            prompt=request.prompt,
            scene_context=request.scene_context,
            private_direction=request.private_direction,
            participants=request.participants,
            conversation_history=request.conversation_history
        )

        return GenerateResponse(
            next_speaker=next_speaker,
            response_text=response_text,
            mood=mood,
            speaking_image=None,  # Claude doesn't use images
            idle_image=None
        )

    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Claude Agent",
        "character": "Pretentious AI"
    }

if __name__ == "__main__":
    port = int(os.getenv("CLAUDE_AGENT_PORT", 9002))
    print(f"🎭 Starting Claude Agent Service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
