#!/usr/bin/env python3
"""
Enhanced Anthropic Adapter for Scene Manager
Supports system blocks with caching like LAURA's main system
"""

import httpx
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AnthropicAdapter:
    """
    Enhanced adapter that supports system blocks and caching
    Based on LAURA's anthropic_adapter pattern but simplified for scenes
    """

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        """
        Initialize enhanced adapter

        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        if not api_key:
            raise ValueError("API key is required for EnhancedAnthropicAdapter")

        self.api_key = api_key
        self.model = model
        logger.info(f"Initialized AnthropicAdapter with model: {self.model}")

    async def generate_response(self,
                               messages: List[Dict[str, Any]],
                               system_blocks: Optional[List[Dict[str, Any]]] = None,
                               system_prompt: Optional[str] = None,
                               max_tokens: int = 150,
                               temperature: float = 0.9) -> Dict[str, Any]:
        """
        Generate response with support for cached system blocks

        Args:
            messages: Conversation messages
            system_blocks: List of system blocks (with optional cache_control)
            system_prompt: Simple string system prompt (fallback)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with content, usage info, and cache metrics
        """
        try:
            # Prepare headers with beta flag for caching
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.API_VERSION,
                "anthropic-beta": "prompt-caching-2024-07-31",  # Enable caching
                "content-type": "application/json"
            }

            # Build payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            # Add system content
            if system_blocks:
                payload["system"] = system_blocks
                logger.debug(f"Using {len(system_blocks)} system blocks")
            elif system_prompt:
                payload["system"] = system_prompt
                logger.debug(f"Using simple system prompt ({len(system_prompt)} chars)")

            # Make API call with 30 second timeout for Director planning
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload
                )

                if response.status_code != 200:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("error", {}).get("message", "Unknown API error")
                    logger.error(f"🚨 Anthropic API error {response.status_code}: {error_msg}")

                    # Debug the error details
                    logger.error(f"🚨 Full error response: {error_data}")
                    logger.error(f"🚨 Request model: {self.model}")
                    logger.error(f"🚨 System blocks count: {len(system_blocks) if system_blocks else 0}")

                    # For 500 errors, raise exception to stop the scene
                    if response.status_code >= 500:
                        raise Exception(f"Anthropic API server error {response.status_code}: {error_msg}")

                    # NO FALLBACKS - FAIL PROPERLY
                    raise Exception(f"API request failed: {error_msg}")

                # Parse response
                response_data = response.json()

                # Extract content
                content_blocks = response_data.get("content", [])
                text_content = ""

                for block in content_blocks:
                    if block.get("type") == "text":
                        text_content += block.get("text", "")

                # Extract usage info including cache metrics
                usage = response_data.get("usage", {})
                cache_metrics = {
                    "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                    "cache_read_tokens": usage.get("cache_read_input_tokens", 0)
                }

                # Log cache performance
                if cache_metrics["cache_read_tokens"] > 0:
                    cache_hit_rate = (
                        cache_metrics["cache_read_tokens"] /
                        (usage.get("input_tokens", 1))
                    ) * 100
                    logger.info(f"📊 Cache hit! {cache_hit_rate:.1f}% of input from cache "
                              f"({cache_metrics['cache_read_tokens']} tokens)")
                elif cache_metrics["cache_creation_tokens"] > 0:
                    logger.info(f"📝 Cache created: {cache_metrics['cache_creation_tokens']} tokens")

                return {
                    "content": text_content.strip(),
                    "usage": usage,
                    "cache_metrics": cache_metrics,
                    "stop_reason": response_data.get("stop_reason"),
                    "error": None
                }

        except httpx.ConnectError as e:
            logger.error(f"Connection error - cannot reach API: {e}")
            raise e
        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {e}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Request error - type: {type(e).__name__}, message: {str(e)}")
            # Check if it's an API key issue
            if not self.api_key:
                logger.error("API key is missing or empty!")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            # NO FALLBACKS - FAIL PROPERLY
            raise e


    def reset_scene(self):
        """Reset for new scene - no state to clear in adapter"""
        logger.debug("EnhancedAnthropicAdapter reset (no-op)")
        pass