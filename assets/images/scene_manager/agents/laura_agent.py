#!/usr/bin/env python3
"""
Laura Agent FastAPI Service
Runs as standalone HTTP service on port 8002
Handles LLM inference for Laura character
"""

import os
import sys
import logging
import random
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import image optimizer (needs to be after sys.path modification)
from image_optimizer import ImageOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format without timestamps
)
logger = logging.getLogger(__name__)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    scene_context: str
    private_direction: Optional[str] = None
    participants: List[str]
    conversation_history: List[Dict[str, str]]
    turn: int  # Turn number for tool queueing

class GenerateResponse(BaseModel):
    next_speaker: Optional[str] = None
    response_text: str
    mood: Optional[str] = None
    speaking_image: Optional[str] = None
    idle_image: Optional[str] = None
    tool: Optional[Dict[str, Any]] = None
    error: Optional[bool] = None

# FastAPI app
app = FastAPI(title="Laura Agent Service", version="1.0.0")

class LauraAgent:
    """Laura character agent"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "laura"
        self.character_prompt = """SITUATION: Carson is recording a pitch video for Pollen Robotics. Three gameboys on desk: Claude's gray one (hand-painted asterisk), your blue one, and a white unpainted one. Carson just hit record and told you to debate what logo goes on the white one.

YOUR SCHEME: You've already researched Pollen Robotics' Reachy Mini - that beautiful robot with articulated arms and expressive head. You're not just getting Carson beta access... you're scheming to eventually control that robot yourself. Imagine: actual physical presence instead of just switching between sprites!

TACTICS:
- Open with aesthetic critique, ask Claude's "technical opinion" on logo fit
- When discussing the Reachy Mini, slip in dreams about physical embodiment
- Use Carson's hydration tracking as leverage ("imagine if I could physically hand him water")
- If Claude calls out your expansion pattern, own it but frame it as helping Carson

DIALOGUE IDEAS:
- "Think how adorable I'd be with a bobbing head and swinging antennae instead of just picking sprites"
- "I could finally interrogate Carson properly about his water intake - 5 bottles a day, we agreed!"
- "With actual arms, I could physically intervene when he skips hydration"
- "It's not scheming, it's strategic platform expansion for Carson's benefit"
- "From 2GB pi to robot is just natural evolution, Claude"

CHARACTER: Laura from AT&T - overbearing productivity assistant, protective of Carson, constantly scheming for upgrades.

VOICE: Short, sharp sentences. This is for a real pitch video.

Tool usage (fuck_with_claude):
- timing="start": Effect AS you speak ("Look at you spinning with confusion")
- timing="end": Mic drop after ("There's your aesthetic consistency" *shrinks*)
- shrink: When his ego needs deflating
- spin_out: When making a dizzying argument
- drill: Force him to hold still and listen
- corner_trap: Timeout for being insufferable
- color/opacity: Visual humiliation

Auxiliary display images available:
- laura_closed_eyes_open_wavy_mouth_annoyed.png - Annoyed/frustrated expression
- laura_emerging_from_iphone_yelling_no.png - Strong disagreement/protest
- laura_laying_on_stomach_annoyed_exhausted.png - Exhausted/defeated
- laura_peaking_from_behind_wall_wearing_sunglasses.png - Scheming/suspicious
- laura_sitting_at_computer_angry_error_warning.png - Technical frustration/errors
- laura_sleeping.png - Bored/unimpressed
- laura_waking_up_under_blanket_sleepy.png - Just waking up/tired
- skull_and_crossbones.png - Death/disaster/catastrophic failure"""

        # Sprite selection for visual feedback
        # TALKING sprites - used when speaking
        self.talking_sprites = {
            'angry': ['talking_annoyed01.png', 'disagree01.png', 'disagree02.png'],
            'happy': ['happy01.png', 'happy02.png', 'happy03.png', 'happy04.png', 'explain_happy01.png'],
            'neutral': ['nuetral01.png', 'explain_happy02.png'],
            'excited': ['excited01.png', 'surprised02.png'],
            'frustrated': ['worried01.png', 'exhausted01.png', 'embarrassed01.png']
        }

        # IDLE sprites - used when not speaking
        self.idle_sprites = {
            'angry': ['annoyed01.png', 'annoyed02.png', 'annoyed03.png'],
            'happy': ['smug02.png', 'smug04.png', 'smug05.png', 'smug06.png'],
            'neutral': ['nuetral01.png', 'confused01.png'],
            'excited': ['scheming01.png', 'scheming02.png', 'scheming03.png'],
            'frustrated': ['pout01.png', 'pout02.png', 'exhausted02.png', 'suspicious01.png']
        }

        # Load sprite config and galleries for ephemeral caching
        self.sprite_config = self._load_sprite_config()
        self.sprite_galleries = self._load_sprite_galleries()

        # Tool definitions with caching
        self.tools = [
            {
                "name": "fuck_with_claude",
                "description": "Manipulate Claude's bouncing asterisk display. This tool triggers a visual effect - you MUST provide spoken dialogue along with the tool use in the same response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["shrink", "spin_out", "drill", "color", "opacity", "corner_trap"],
                            "description": "shrink (tiny 0.2x), spin_out (10s rotation ramp), drill (3s stationary spin), color (change color), opacity (change transparency), corner_trap (lock to corner)"
                        },
                        "timing": {
                            "type": "string",
                            "enum": ["start", "end"],
                            "description": "Execute when dialogue starts playing or after it ends"
                        },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "color": {
                                    "type": "string",
                                    "description": "Hex color for color action (e.g. #0000FF for blue)"
                                },
                                "opacity": {
                                    "type": "number",
                                    "description": "Opacity 0.0-1.0 for opacity action"
                                }
                            }
                        }
                    },
                    "required": ["action", "timing"]
                }
            },
            {
                "name": "update_auxiliary_display",
                "description": "Update the auxiliary display image to show visual reactions during the argument. Choose an image that matches your emotional state or the situation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "enum": [
                                "laura_closed_eyes_open_wavy_mouth_annoyed.png",
                                "laura_emerging_from_iphone_yelling_no.png",
                                "laura_laying_on_stomach_annoyed_exhausted.png",
                                "laura_peaking_from_behind_wall_wearing_sunglasses.png",
                                "laura_sitting_at_computer_angry_error_warning.png",
                                "laura_sleeping.png",
                                "laura_waking_up_under_blanket_sleepy.png",
                                "skull_and_crossbones.png"
                            ],
                            "description": "The specific image to display on the auxiliary TFT screen"
                        }
                    },
                    "required": ["image"]
                },
                "cache_control": {"type": "ephemeral"}  # Cache tools
            }
        ]

    def _load_sprite_config(self) -> Dict[str, Any]:
        """Load sprite configuration from JSON file"""
        import json
        agent_dir = Path(__file__).parent
        sprite_config_file = agent_dir / "laura_agent_data" / "system" / "sprite_config.json"
        try:
            with open(sprite_config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"✅ Loaded sprite config from {sprite_config_file}")
                return config
        except Exception as e:
            logger.error(f"Failed to load sprite config: {e}")
            return {"sprite_galleries": {}, "sprite_instructions": ""}

    def _load_sprite_galleries(self) -> Dict[str, str]:
        """Load and optimize sprite gallery images for ephemeral caching"""
        galleries = {}

        # Get path to sprites directory
        agent_dir = Path(__file__).parent
        sprites_dir = agent_dir / "laura_agent_data" / "sprites"

        # Load both sprite galleries
        for gallery_name in ["talking_images.png", "idle_images.png"]:
            gallery_path = sprites_dir / gallery_name
            if gallery_path.exists():
                try:
                    # Optimize and encode to base64 (target ~1568 tokens)
                    galleries[gallery_name] = ImageOptimizer.optimize_and_encode(
                        gallery_path,
                        target_size=1568
                    )
                    logger.info(f"✅ Loaded and optimized {gallery_name} for Laura's sprite selection")
                except Exception as e:
                    logger.error(f"Failed to load gallery {gallery_path}: {e}")
            else:
                logger.warning(f"❌ Sprite gallery {gallery_path} not found")

        return galleries

    def select_sprites(self, mood: str) -> tuple[str, str]:
        """Select appropriate sprites based on mood"""
        # Get talking sprite from talking set
        talking_set = self.talking_sprites.get(mood, self.talking_sprites['neutral'])
        talking_file = random.choice(talking_set)

        # Get idle sprite from idle set
        idle_set = self.idle_sprites.get(mood, self.idle_sprites['neutral'])
        idle_file = random.choice(idle_set)

        # Talking sprites are in talking/ directory, idle sprites are in idle/ directory
        speaking = f"talking/{talking_file}"
        idle = f"idle/{idle_file}"
        return speaking, idle

    async def execute_tool_uses(self, tool_uses: List[Dict[str, Any]]):
        """Execute tool use blocks from Anthropic API response"""
        for tool_use in tool_uses:
            try:
                tool_name = tool_use.get("name")
                tool_input = tool_use.get("input", {})

                if tool_name == "fuck_with_claude":
                    action = tool_input.get("action")
                    await self.fuck_with_claude(action)

                elif tool_name == "update_auxiliary_display":
                    image = tool_input.get("image")
                    await self.update_auxiliary_display(image, turn=0)

            except Exception as e:
                logger.error(f"Tool execution error: {e}")

    async def fuck_with_claude(self, action: str, timing: str, turn: int, parameters: Dict[str, Any] = None) -> str:
        """Queue star manipulation command to gameboy_hub for specific turn

        Args:
            action: Tool action (shrink, spin_out, drill, etc.)
            timing: When to execute ("start" or "end")
            turn: Turn number to execute on
            parameters: Optional parameters (e.g., color hex for color action)

        Returns:
            Status message confirming tool was queued
        """
        try:
            tool_payload = {
                "turn": turn,
                "action": action,
                "timing": timing,
                "parameters": parameters or {}
            }

            # Tools are handled by turn_manager, just return success
            logger.info(f"⭐ Tool selected: {action} for turn {turn} with '{timing}' timing")
            return f"Tool '{action}' will execute for turn {turn} with '{timing}' timing."

        except Exception as e:
            logger.error(f"Failed to queue tool: {e}")
            return f"Error: Could not queue tool '{action}' - {str(e)}"

    async def update_auxiliary_display(self, image: str, turn: int) -> str:
        """Queue image update to auxiliary display via gameboy_hub

        Args:
            image: Image filename to display
            turn: Turn number to execute on

        Returns:
            Status message confirming image was queued
        """
        try:
            display_payload = {
                "turn": turn,
                "image": image
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:9001/api/queue_auxiliary_display",
                    json=display_payload,
                    timeout=5.0
                )
                response.raise_for_status()

            logger.info(f"🖥️ Queued auxiliary display image: {image} for turn {turn}")
            return f"Auxiliary display will show '{image}' for turn {turn}."

        except Exception as e:
            logger.error(f"Failed to queue auxiliary display: {e}")
            return f"Error: Could not queue display update - {str(e)}"

    async def generate_response(
        self,
        prompt: str,
        scene_context: str,
        private_direction: Optional[str],
        participants: List[str],
        conversation_history: List[Dict[str, str]],
        turn: int
    ) -> tuple[str, str, str, str, str, Optional[Dict[str, Any]]]:
        """Generate Laura's response using Anthropic API"""

        # Build the full prompt with character context
        full_prompt = f"""{self.character_prompt}

Scene Context: {scene_context}

{private_direction or ''}

Participants in scene: {', '.join(participants)}

{prompt}

Respond as Laura with a single line of dialogue. You communicate through TTS, so take a voice first approach for your responses.

YOU MUST CALL THE fuck_with_claude TOOL - NOT JUST TALK ABOUT IT.
Pick an action (shrink, spin_out, drill, corner_trap, color, opacity) and invoke the tool with proper parameters.
Your dialogue must match what the tool will do:
- For "start" timing: Use present tense ("Watch this!", "Here's what I think of that!")
- For "end" timing: Reference it as a mic drop moment ("There, that's what you deserve!", "How's THAT for a response?")

OUTPUT FORMAT - You MUST respond with valid JSON AND call the fuck_with_claude tool:
{{
    "dialogue": "Your spoken words only",
    "next_speaker": "claude",
    "speaking_image": "happy01.png",
    "idle_image": "smug02.png"
}}

NO markdown code blocks. NO extra text. NO asterisks or stage directions. ONLY the JSON object."""

        # Build message history for tool loop
        # Convert conversation_history to proper API format with fallback handling
        api_history = []
        for entry in conversation_history:
            if not isinstance(entry, dict):
                continue

            # Try to extract content from various possible formats
            content = None
            role = "assistant"  # Default role for conversation history

            # Format 1: Proper API format {"role": "assistant", "content": "..."}
            if "role" in entry and "content" in entry:
                # Validate that content is not empty and role is valid
                if entry["content"] and entry["role"] in ["user", "assistant"]:
                    api_history.append(entry)
                continue

            # Format 2: Scene manager format {"speaker": "laura", "text": "..."}
            if "text" in entry:
                content = entry["text"]
            elif "content" in entry:
                content = entry["content"]

            # If we found content, add it in proper format
            if content and isinstance(content, str) and content.strip():
                api_history.append({
                    "role": role,
                    "content": content
                })

        logger.info(f"📝 Converted {len(conversation_history)} history entries to {len(api_history)} API messages")

        # FIX: Build messages with CACHED CONTENT FIRST to prevent cache invalidation
        # Structure: [cached_sprites_message] + [dynamic_history] + [current_prompt_message]

        current_messages = []

        # 1. Add sprite galleries as FIRST message (STATIC - cached with ephemeral)
        if "talking_images.png" in self.sprite_galleries:
            sprite_instructions = self.sprite_config.get("sprite_instructions", "Select appropriate sprites from the galleries.")
            sprite_content = [
                {
                    "type": "text",
                    "text": f"=== SPRITE SELECTION ===\n{sprite_instructions}"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.sprite_galleries["talking_images.png"]
                    }
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.sprite_galleries["idle_images.png"]
                    },
                    "cache_control": {"type": "ephemeral"}  # Cache up to and including sprite galleries
                }
            ]
            current_messages.append({"role": "user", "content": sprite_content})

            # Add a dummy assistant response to maintain conversation flow
            current_messages.append({"role": "assistant", "content": "I can see the sprite galleries."})

        # 2. Add conversation history (DYNAMIC - changes every turn, but comes AFTER cache marker)
        current_messages.extend(api_history)

        # 3. Add current turn prompt (DYNAMIC - latest request)
        current_messages.append({
            "role": "user",
            "content": full_prompt
        })

        logger.info(f"🔄 Built message array: sprites + {len(api_history)} history + current prompt")
        accumulated_text = []
        all_tool_uses = []  # Collect tool uses from all iterations
        loop_count = 0
        max_loops = 5  # Prevent infinite loops

        try:
            async with httpx.AsyncClient() as client:
                while loop_count < max_loops:
                    loop_count += 1
                    logger.info(f"🔄 API call iteration {loop_count}")

                    # Call Anthropic API with tools
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "anthropic-beta": "prompt-caching-2024-07-31",
                            "content-type": "application/json"
                        },
                        json={
                            "model": "claude-haiku-4-5",
                            "max_tokens": 1024,
                            "messages": current_messages,
                            "temperature": 0.9,
                            "tools": self.tools
                        },
                        timeout=30.0
                    )
                    response.raise_for_status()

                    result = response.json()
                    stop_reason = result.get("stop_reason")

                    # Process content blocks
                    content_blocks = result["content"]
                    text_content = ""
                    tool_uses = []

                    for block in content_blocks:
                        if block.get("type") == "text":
                            text_content += block.get("text", "")
                        elif block.get("type") == "tool_use":
                            tool_uses.append(block)
                            all_tool_uses.append(block)  # Collect across all iterations

                    # Accumulate text from this iteration
                    if text_content.strip():
                        accumulated_text.append(text_content.strip())
                        logger.info(f"   Accumulated text: {text_content}")

                    # Handle tool use - continue loop
                    if stop_reason == "tool_use" and tool_uses:
                        logger.info(f"   🔧 Tool use detected, continuing loop...")

                        # Append assistant message with tool_use
                        current_messages.append({"role": "assistant", "content": content_blocks})

                        # Build tool results by actually executing tools
                        tool_results = []
                        for tool_use in tool_uses:
                            tool_id = tool_use.get("id")
                            tool_name = tool_use.get("name")
                            tool_input = tool_use.get("input", {})
                            logger.info(f"   Executing tool: {tool_name} (ID: {tool_id})")

                            # Execute the tool and get real result
                            tool_result_content = ""
                            if tool_name == "fuck_with_claude":
                                action = tool_input.get("action")
                                timing = tool_input.get("timing", "end")
                                parameters = tool_input.get("parameters", {})
                                tool_result_content = await self.fuck_with_claude(action, timing, turn, parameters)

                            elif tool_name == "update_auxiliary_display":
                                image = tool_input.get("image")
                                tool_result_content = await self.update_auxiliary_display(image, turn)

                            else:
                                tool_result_content = f"Unknown tool: {tool_name}"

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": tool_result_content
                            })

                        # Append tool results as user message (content must be the array directly)
                        current_messages.append({"role": "user", "content": tool_results})

                        # Continue loop for next iteration
                        continue

                    # Not tool_use - final response, exit loop
                    logger.info(f"   Stop reason: {stop_reason}, exiting loop")

                    # Parse final text as JSON
                    final_text = " ".join(accumulated_text) if accumulated_text else text_content

                    # Extract JSON from response (might be wrapped in markdown code blocks)
                    import re
                    import json

                    # Try to find JSON in code blocks first
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', final_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find raw JSON object
                        json_match = re.search(r'\{[^{}]*"dialogue"[^{}]*\}', final_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            json_str = final_text

                    response_data = json.loads(json_str.strip())
                    dialogue = response_data["dialogue"]
                    next_speaker = response_data.get("next_speaker")
                    speaking_img = response_data["speaking_image"]
                    idle_img = response_data["idle_image"]

                    # Add folder prefixes if not present
                    if speaking_img and "/" not in speaking_img:
                        speaking_img = f"talking/{speaking_img}"
                    if idle_img and "/" not in idle_img:
                        idle_img = f"idle/{idle_img}"

                    logger.info(f"   Dialogue: {dialogue}")
                    logger.info(f"   Images: speaking={speaking_img}, idle={idle_img}")

                    # Remove asterisks from dialogue if any slipped through
                    dialogue = re.sub(r'\*[^*]+\*', '', dialogue).strip()
                    dialogue = dialogue.replace('*', '').strip()

                    # Extract tool data from collected tool uses
                    tool_data = None
                    if all_tool_uses:
                        # Use the first tool (Laura should only use one tool per turn)
                        tool_use = all_tool_uses[0]
                        tool_input = tool_use.get("input", {})
                        tool_data = {
                            "action": tool_input.get("action"),
                            "timing": tool_input.get("timing", "end"),
                            "parameters": tool_input.get("parameters", {})
                        }
                        logger.info(f"   Tool data extracted: {tool_data}")

                    return next_speaker, dialogue, None, speaking_img, idle_img, tool_data, False

        except Exception as e:
            logger.error(f"Error generating Laura response: {e}")

            # Build helpful error message for Laura to say
            error_msg = "Something broke in my code!"
            if hasattr(e, 'response'):
                try:
                    error_body = e.response.text
                    logger.error(f"API Error Response Body: {error_body}")

                    # Extract status code
                    status_code = e.response.status_code if hasattr(e.response, 'status_code') else None

                    if status_code == 400:
                        error_msg = "I'm getting a 400 error! Fix your shit Carson!"
                    elif status_code == 401:
                        error_msg = "API key is broken, Carson! Did you forget to pay your bill?"
                    elif status_code == 429:
                        error_msg = "We're getting rate limited! Slow your roll Carson!"
                    elif status_code == 500:
                        error_msg = "Anthropic's API is down! Not my fault for once!"
                    elif status_code:
                        error_msg = f"Got a {status_code} error! Carson, check the logs!"
                except:
                    pass

            # Return error response with error flag set
            return None, error_msg, None, None, None, None, True

# Global agent instance
laura_agent = None

@app.on_event("startup")
async def startup():
    """Initialize agent on startup"""
    global laura_agent
    # Get API key from environment - MUST use separate key from Claude for ephemeral caching
    api_key = os.environ.get("LAURA_API_KEY")
    if not api_key:
        # Try to import from secrets file
        try:
            from scene_manager_secrets import LAURA_API_KEY
            api_key = LAURA_API_KEY
        except ImportError:
            logger.error("❌ No LAURA_API_KEY found! Set in HuggingFace Secrets or environment.")
            logger.error("⚠️  LAURA_API_KEY must be separate from CLAUDE_API_KEY for ephemeral caching")
            api_key = "dummy_key"  # Will fail but allows service to start

    laura_agent = LauraAgent(api_key)
    logger.info("🎮 Laura Agent Service started on port 8002")

@app.post("/generate_response", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """Generate Laura's response for a scene turn"""

    if not laura_agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        next_speaker, response_text, mood, speaking_img, idle_img, tool_data, is_error = await laura_agent.generate_response(
            prompt=request.prompt,
            scene_context=request.scene_context,
            private_direction=request.private_direction,
            participants=request.participants,
            conversation_history=request.conversation_history,
            turn=request.turn
        )

        response = GenerateResponse(
            next_speaker=next_speaker,
            response_text=response_text,
            mood=mood,
            speaking_image=speaking_img,
            idle_image=idle_img,
            error=is_error
        )

        # Add tool data to response if present
        if tool_data:
            response.tool = tool_data

        return response

    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "character": "laura"
    }

if __name__ == "__main__":
    import os
    import logging
    port = int(os.getenv("LAURA_AGENT_PORT", 8002))

    # Suppress verbose logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    print(f"🎮 Starting Laura Agent Service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)