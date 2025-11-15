#!/usr/bin/env python3
"""
Image Optimizer for Scene Manager
Resizes images to appropriate sizes for Anthropic API (max 1568px, min 200px)
"""

import base64
import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ImageOptimizer:
    """Optimizes images for LLM vision APIs"""

    @staticmethod
    def resize_image(img: Image.Image, max_dimension: int = 1568, min_dimension: int = 200) -> Image.Image:
        """
        Resize image to fit within API constraints
        - Max 1568px on longest side
        - Min 200px on shortest side
        - Maintains aspect ratio
        """
        width, height = img.size
        aspect_ratio = width / height

        # Check if image needs resizing
        needs_resize = False
        new_width, new_height = width, height

        # Handle too large (max 1568px on longest side)
        if width > max_dimension or height > max_dimension:
            needs_resize = True
            if width > height:
                new_width = max_dimension
                new_height = int(max_dimension / aspect_ratio)
            else:
                new_height = max_dimension
                new_width = int(max_dimension * aspect_ratio)

        # Handle too small (min 200px on shortest side)
        elif width < min_dimension or height < min_dimension:
            needs_resize = True
            if width < height:
                new_width = min_dimension
                new_height = int(min_dimension / aspect_ratio)
            else:
                new_height = min_dimension
                new_width = int(min_dimension * aspect_ratio)

        if needs_resize:
            logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            return img.resize((new_width, new_height), Image.LANCZOS)
        return img

    @staticmethod
    def optimize_and_encode(image_path: Path, target_size: Optional[int] = 768) -> str:
        """
        Load, resize, optimize and base64 encode an image

        Args:
            image_path: Path to image file
            target_size: Target size for longest dimension (default 768 for faster loading)

        Returns:
            Base64 encoded optimized image
        """
        try:
            with Image.open(image_path) as img:
                # Convert RGBA to RGB if needed
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize to target size
                if target_size:
                    img = ImageOptimizer.resize_image(img, max_dimension=target_size)

                # Convert to JPEG for smaller size (PNG for sprite galleries with transparency)
                output_format = 'PNG' if 'sprite' in str(image_path).lower() or 'idle' in str(image_path).lower() or 'talking' in str(image_path).lower() else 'JPEG'

                # Save to bytes
                img_byte_arr = io.BytesIO()
                if output_format == 'JPEG':
                    img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                else:
                    img.save(img_byte_arr, format='PNG', optimize=True)

                img_byte_arr.seek(0)

                # Encode to base64
                encoded = base64.b64encode(img_byte_arr.read()).decode('utf-8')

                # Log size reduction
                original_size = image_path.stat().st_size
                encoded_size = len(encoded)
                reduction = (1 - encoded_size / (original_size * 1.37)) * 100  # Base64 is ~1.37x larger
                logger.debug(f"Optimized {image_path.name}: {original_size:,} -> {encoded_size:,} bytes ({reduction:.1f}% reduction)")

                return encoded

        except Exception as e:
            logger.error(f"Failed to optimize image {image_path}: {e}")
            # Fallback to raw loading
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    @staticmethod
    def get_media_type(filename: str, optimized: bool = True) -> str:
        """Get MIME type for image"""
        ext = Path(filename).suffix.lower()

        if optimized:
            # We convert most images to JPEG for size
            if 'sprite' in filename.lower() or 'idle' in filename.lower() or 'talking' in filename.lower():
                return 'image/png'
            return 'image/jpeg'
        else:
            # Original format
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif'
            }
            return mime_types.get(ext, 'image/jpeg')