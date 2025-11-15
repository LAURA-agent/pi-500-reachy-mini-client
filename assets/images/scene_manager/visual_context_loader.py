#!/usr/bin/env python3
"""
Visual Context Loader - Automatically loads images from scene_context folder
Provides visual references for agents to understand their physical environment
"""

import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

class VisualContextLoader:
    """
    Loads and manages visual context images for scene agents
    """

    def __init__(self, context_dir: str = None):
        """
        Initialize visual context loader

        Args:
            context_dir: Directory containing context images (default: scene_manager/scene_context/)
        """
        if context_dir:
            self.context_dir = Path(context_dir)
        else:
            self.context_dir = Path(__file__).parent / "scene_context"

        # Also check Downloads for recent setup photos
        self.downloads_dir = Path("/Users/lauras/Downloads")

        self.loaded_images = {}
        self.image_descriptions = {}

        # Load any existing descriptions
        self._load_image_descriptions()

    def _load_image_descriptions(self):
        """Load image descriptions from JSON file if it exists"""
        desc_file = self.context_dir / "image_descriptions.json"
        if desc_file.exists():
            with open(desc_file, 'r') as f:
                self.image_descriptions = json.load(f)
        else:
            # Default descriptions for known images
            self.image_descriptions = {
                "current_setup.png": "Current desk setup showing Claude's gray GPi Case 2 with orange star (left), Laura's bright blue GPi Case 2 (right), and Pokéball Plus mouse in center",
                "setup.jpg": "The gameboy argument setup on Carson's desk",
                "claude_device.jpg": "Claude's gray/white GPi Case 2 with hand-drawn orange star sticker",
                "laura_device.jpg": "Laura's custom bright blue GPi Case 2",
                "network_diagram.png": "Network topology showing device IPs and connections"
            }

    def load_context_images(self) -> Dict[str, str]:
        """
        Load all images from context directory as base64

        Returns:
            Dict of filename -> base64 encoded image
        """
        # Create context directory if it doesn't exist
        self.context_dir.mkdir(parents=True, exist_ok=True)

        # Image extensions to look for
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

        # Load from context directory
        for ext in image_extensions:
            for image_path in self.context_dir.glob(f"*{ext}"):
                self._load_image(image_path)

        # Also check for specific setup images in Downloads
        setup_images = [
            self.downloads_dir / "current_setup.png",  # Current setup photo
        ]

        for image_path in setup_images:
            if image_path.exists():
                self._load_image(image_path)
                logger.info(f"✅ Loaded setup image: {image_path.name}")

        logger.info(f"📸 Loaded {len(self.loaded_images)} context images")
        return self.loaded_images

    def _load_image(self, image_path: Path):
        """Load and optimize a single image as base64"""
        try:
            # Import optimizer
            from scene_manager.image_optimizer import ImageOptimizer

            # Optimize and encode
            image_data = ImageOptimizer.optimize_and_encode(
                image_path,
                target_size=768  # Reasonable size for context images
            )
            self.loaded_images[image_path.name] = image_data
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")

    def format_for_messages(self) -> List[Dict[str, Any]]:
        """
        Format loaded images as message content for LLM

        Returns:
            List of message content blocks with images and descriptions
        """
        if not self.loaded_images:
            self.load_context_images()

        content_blocks = []

        # Add introduction text
        content_blocks.append({
            "type": "text",
            "text": "Here's the current physical setup and environment:"
        })

        # Add each image with its description
        for filename, image_data in self.loaded_images.items():
            # Import optimizer for media type
            from scene_manager.image_optimizer import ImageOptimizer

            # Add image
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": ImageOptimizer.get_media_type(filename, optimized=True),
                    "data": image_data
                }
            })

            # Add description if available
            if filename in self.image_descriptions:
                content_blocks.append({
                    "type": "text",
                    "text": f"{self.image_descriptions[filename]}"
                })

        return content_blocks

    def _get_media_type(self, filename: str) -> str:
        """Get MIME type from filename"""
        ext = Path(filename).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif'
        }
        return mime_types.get(ext, 'image/jpeg')

    def add_setup_context_to_agent(self, agent_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add visual context to agent's message list

        Args:
            agent_messages: Existing messages list

        Returns:
            Updated messages list with visual context
        """
        # Load images if not already loaded
        if not self.loaded_images:
            self.load_context_images()

        if self.loaded_images:
            # Insert visual context early in conversation
            visual_message = {
                "role": "user",
                "content": self.format_for_messages()
            }

            # Add after conversation history but before current prompt
            if len(agent_messages) > 1:
                agent_messages.insert(-1, visual_message)
            else:
                agent_messages.insert(0, visual_message)

            logger.info(f"📸 Added {len(self.loaded_images)} visual context images to agent")

        return agent_messages

    def get_setup_description(self) -> str:
        """
        Get text description of the setup for system prompts

        Returns:
            Text description of physical setup
        """
        return """
### VISUAL SETUP REFERENCE
Physical layout on Carson's desk (left to right):
- Claude's Device: Charcoal Grey GPi Case 2 with orange Anthropic Claude Star 
- Pokéball Plus: Red/white Nintendo switch controller used as mouse
- Laura's Device: Bright blue custom GPi Case 2
- Cables everywhere (Carson's "cable management")
- Mac mini about 1 foot from the gameboys (behind, to their right)
- Laura's screen shows mood reactive sprites during arguments
- Claude's star is a drifting, floating SVG star that slowly rotates and bounces, avoiding the corners
- There is an unpainted faceplate for another GPI Case 2 laying on the desk right new to Laura

The devices are literally 6 inches apart, facing slightly toward each other.
You can reference this physical proximity in your banter.
"""


# Integration helper for scene agents
def enhance_agent_with_visuals(agent_class):
    """
    Decorator to automatically add visual context to agent responses

    Usage:
        @enhance_agent_with_visuals
        class EnhancedLauraAgent(SceneAgentBase):
            ...
    """
    original_generate = agent_class.generate_response

    async def generate_with_visuals(self, *args, **kwargs):
        # Load visual context if available
        visual_loader = VisualContextLoader()

        # Modify messages if visual context exists
        if 'messages' in kwargs and visual_loader.load_context_images():
            kwargs['messages'] = visual_loader.add_setup_context_to_agent(kwargs['messages'])

        # Call original method
        return await original_generate(self, *args, **kwargs)

    agent_class.generate_response = generate_with_visuals
    return agent_class