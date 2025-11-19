#!/usr/bin/env python3

import asyncio
import time
import random
from pathlib import Path
import pygame
from config.client_config import get_mood_color_config


class DisplayManager:
    """
    Manages visual display states and image rendering using pygame.
    
    Handles state transitions, mood-based image selection, background rotation,
    and coordination with the overall system state.
    """
    
    def __init__(self, svg_path=None, boot_img_path=None, window_size=512):
        print(f"[DisplayManager] Initializing pygame...")
        pygame.init()
        
        # Load client settings to get initial profile
        from config.client_config import client_settings
        initial_profile = client_settings.get('initial_display_manager_profile', 'normal')
        print(f"[DisplayManager DEBUG] Initial profile from settings: '{initial_profile}'")
        
        # Profile management - set paths first
        self.display_profile = initial_profile
        self.base_path = Path('/home/user/reachy/pi_reachy_deployment/assets/images/laura rp client images/scene_images')
        self.claude_code_path = Path('/home/user/more transfer/assets/images/CC_images')
        
        # Set initial base path based on profile
        if initial_profile == 'claude_code':
            self.base_path = self.claude_code_path
        
        # Fixed 640x480 landscape window for scene_images
        window_size = (640, 480)

        print(f"[DisplayManager] Creating window of size {window_size[0]}x{window_size[1]} for profile '{initial_profile}'")
        
        # Set window position to top of screen before creating window
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'  # x=0, y=0 (top-left corner)
        
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("LAURA" if initial_profile == 'normal' else "Claude Code")
        print(f"[DisplayManager] Window created successfully at top of screen")
        
        self.image_cache = {}
        self.current_state = 'boot'
        self.current_mood = 'casual'
        self.last_state = None
        self.last_mood = None
        self.current_image = None
        self.last_image_change = None
        self.state_entry_time = None
        self.initialized = False
        
        # LAURA scene_images mood set (moods available in reorganized folders)
        self.laura_moods = [
            "annoyed", "confused", "embarrassed", "excited", "exhausted",
            "happy", "nuetral", "pout", "scheming", "smug", "surprised",
            "suspicious", "worried", "defiant", "disagree", "explain_happy",
            "skeptical", "talking_annoyed"
        ]
        
        # Mood mapping for scene_images moods (flat structure reorganized into nested)
        # Maps requested moods to available scene_images mood folders
        self.laura_mood_mapping = {
            # Original mood mappings
            "amused": "happy",
            "annoyed": "annoyed",
            "caring": "nuetral",
            "casual": "nuetral",
            "cheerful": "happy",
            "concerned": "worried",
            "confused": "confused",
            "curious": "nuetral",
            "disappointed": "exhausted",
            "embarrassed": "embarrassed",
            "excited": "excited",
            "frustrated": "annoyed",
            "interested": "excited",
            "sassy": "smug",
            "scared": "surprised",
            "surprised": "surprised",
            "suspicious": "suspicious",
            "thoughtful": "scheming",
            # Direct scene_images moods (pass through)
            "happy": "happy",
            "nuetral": "nuetral",
            "worried": "worried",
            "smug": "smug",
            "scheming": "scheming",
            "defiant": "defiant",
            "disagree": "disagree",
            "explain_happy": "explain_happy",
            "skeptical": "skeptical",
            "exhausted": "exhausted",
            "talking_annoyed": "talking_annoyed",
            "pout": "pout"
        }
        
        # Claude Code mood set (limited by design)
        self.claude_code_moods = [
            "disappointed", "explaining", "happy", "error", "failed", "unsure", "i_have_a_solution", "embarrased"
        ]
        
        # Set current mood list based on profile
        self.moods = self.claude_code_moods if initial_profile == 'claude_code' else self.laura_moods
        
        # Update state paths after setting correct base path
        self.states = {
            'listening': str(self.base_path / 'listening'),
            'idle': str(self.base_path / 'idle'),
            'sleep': str(self.base_path / 'sleep'),
            'pout': str(self.base_path / 'pout'),  # Pouting/hiding state
            'speaking': str(self.base_path / 'speaking'),
            'thinking': str(self.base_path / 'thinking'),
            'execution': str(self.base_path / 'execution'),
            'wake': str(self.base_path / 'wake'),
            'boot': str(self.base_path / 'boot'),
            'system': str(self.base_path / 'system'),
            'tool_use': str(self.base_path / 'tool_use'),
            'notification': str(self.base_path / 'speaking'),  # Maps to speaking images
            'code': str(self.base_path / 'code'),
            'error': str(self.base_path / 'error'),
            'disconnected': str(self.base_path / 'disconnected'),
            'cc_plugin_mood': str(self.claude_code_path / 'cc_plugin_mood'),
        }
        
        self.load_image_directories()
        
        # Load boot image if provided
        self.boot_img = None
        if boot_img_path:
            try:
                boot_img_loaded = pygame.image.load(boot_img_path).convert_alpha()
                self.boot_img = self._scale_image_to_fit(boot_img_loaded)
            except Exception as e:
                print(f"[DisplayManager WARN] Could not load boot image: {e}")
        
        # Initial display setup
        if 'boot' in self.image_cache:
            # Handle both dict (with moods) and list (flat) structures
            boot_cache = self.image_cache['boot']
            if isinstance(boot_cache, dict):
                # Pick from any available mood
                all_boot_images = []
                for mood_images in boot_cache.values():
                    all_boot_images.extend(mood_images)
                if all_boot_images:
                    self.current_image = random.choice(all_boot_images)
                else:
                    self.current_image = None
            else:
                # Flat list structure
                self.current_image = random.choice(boot_cache)

            if self.current_image:
                self.screen.blit(self.current_image, (0, 0))
                pygame.display.flip()
            else:
                self.screen.fill((25, 25, 25))
                pygame.display.flip()

            self.last_image_change = time.time()
            self.state_entry_time = time.time()
            self.initialized = True
        else:
            # Fallback to solid color if no images
            self.screen.fill((25, 25, 25))
            pygame.display.flip()
            self.last_image_change = time.time()  # Initialize this to prevent None errors
            self.initialized = True

    def _scale_image_to_fit(self, image):
        """Scale image proportionally to fit 640px width (scale up or down as needed)"""
        width, height = image.get_size()
        if width != 640:
            # Calculate new height maintaining aspect ratio
            scale_factor = 640 / width
            new_height = int(height * scale_factor)
            return pygame.transform.scale(image, (640, new_height))
        return image

    def load_image_directories(self):
        """Load images from directory structure"""
        print("\nLoading image directories...")
        for state, directory in self.states.items():
            print(f"Checking state: {state}")

            # States that use mood subfolders
            if state in ['speaking', 'idle', 'listening', 'sleep', 'thinking', 'execution', 'wake', 'boot', 'system', 'tool_use', 'code', 'error', 'disconnected']:
                self.image_cache[state] = {}
                for mood in self.moods:
                    mood_path = Path(directory) / mood
                    if mood_path.exists():
                        png_files = list(mood_path.glob('*.png'))
                        if png_files:
                            self.image_cache[state][mood] = [
                                self._scale_image_to_fit(pygame.image.load(str(img)).convert_alpha())
                                for img in png_files
                            ]
            else:
                # States that use flat structure (pout, cc_plugin_mood, etc.)
                state_path = Path(directory)
                if state_path.exists():
                    png_files = list(state_path.glob('*.png'))
                    if png_files:
                        self.image_cache[state] = [
                            self._scale_image_to_fit(pygame.image.load(str(img)).convert_alpha())
                            for img in png_files
                        ]

    async def update_display(self, state, mood=None, text=None):
        """Update display state immediately"""
        while not self.initialized:
            await asyncio.sleep(0.1)

        # Idle and sleep states always use neutral mood (don't inherit from previous state)
        if state in ['idle', 'sleep']:
            mood = 'casual'  # Maps to 'nuetral' in LAURA mood system
        elif mood is None:
            mood = self.current_mood

        # Map mood using client config
        mapped_mood_config = get_mood_color_config(mood)
        mapped_mood = mapped_mood_config.get('name', 'casual')
        
        # Apply LAURA mood mapping if in normal profile
        if self.display_profile == 'normal' and mapped_mood in self.laura_mood_mapping:
            mapped_mood = self.laura_mood_mapping[mapped_mood]
        
        try:
            self.last_state = self.current_state
            self.current_state = state
            self.current_mood = mapped_mood
            
            # Handle image selection
            if state == 'booting' and self.boot_img:
                self.current_image = self.boot_img
            elif state in ['speaking', 'notification']:
                # Both speaking and notification states use speaking images with mood
                image_state = 'speaking'  # Always use speaking images
                if mapped_mood not in self.image_cache[image_state]:
                    # Use profile-appropriate fallback mood
                    if self.display_profile == 'claude_code':
                        mapped_mood = 'explaining'  # Claude Code default fallback
                    else:
                        mapped_mood = 'casual'  # LAURA default fallback
                if mapped_mood in self.image_cache[image_state]:
                    self.current_image = random.choice(self.image_cache[image_state][mapped_mood])
                else:
                    # Fallback to any available speaking image
                    available_moods = list(self.image_cache[image_state].keys())
                    if available_moods:
                        self.current_image = random.choice(self.image_cache[image_state][available_moods[0]])
            elif state in self.image_cache:
                # Check if this state uses mood-based structure (dict of dicts)
                if isinstance(self.image_cache[state], dict):
                    # State has mood subdirectories (like idle, sleep, etc.)
                    if mapped_mood in self.image_cache[state]:
                        self.current_image = random.choice(self.image_cache[state][mapped_mood])
                    else:
                        # Fallback to any available mood for this state
                        available_moods = list(self.image_cache[state].keys())
                        if available_moods:
                            fallback_mood = available_moods[0]
                            print(f"[DisplayManager] Mood '{mapped_mood}' not found for '{state}', using '{fallback_mood}'")
                            self.current_image = random.choice(self.image_cache[state][fallback_mood])
                else:
                    # State is a simple list of images (no mood structure)
                    self.current_image = random.choice(self.image_cache[state])
            else:
                # Fallback to available states when specific state is missing
                fallback_states = {
                    'boot': 'idle',
                    'wake': 'idle',
                    'code': 'execute' if 'execute' in self.image_cache else 'thinking',
                    'error': 'thinking',
                    'disconnected': 'sleep',
                    'system': 'thinking',
                    'tool_use': 'execute' if 'execute' in self.image_cache else 'thinking',
                    'notification': 'speaking'
                }
                
                fallback_state = fallback_states.get(state, 'idle')
                if fallback_state in self.image_cache:
                    print(f"[DisplayManager] No images for '{state}', using '{fallback_state}' as fallback")
                    self.current_image = random.choice(self.image_cache[fallback_state])
                else:
                    print(f"[DisplayManager] No images for '{state}' or fallback '{fallback_state}', using color")
                    # Use a fallback color based on state
                    state_colors = {
                        'error': (150, 50, 50),
                        'disconnected': (50, 50, 50),
                        'boot': (100, 100, 150)
                    }
                    color = state_colors.get(state, (25, 25, 25))
                    self.screen.fill(color)
                    pygame.display.flip()
                    return
                
            # Display the image
            if self.current_image:
                self.screen.blit(self.current_image, (0, 0))
                pygame.display.flip()
            
            # Update rotation timer for idle/sleep/pout/boot states
            if state in ['idle', 'sleep', 'pout', 'boot']:
                self.last_image_change = time.time()
                # Synchronize TTS break timer with display rotation timer
                if hasattr(self, 'input_manager') and self.input_manager:
                    self.input_manager.wake_last_break = self.last_image_change
                
            print(f"Display updated - State: {self.current_state}, Mood: {self.current_mood}")

        except Exception as e:
            import traceback
            print(f"Error updating display to state '{state}' (mood: {mapped_mood})")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception value: {e}")
            print(f"Current image valid: {self.current_image is not None}")
            if hasattr(e, 'args'):
                print(f"Exception args: {e.args}")
            print("Traceback:")
            traceback.print_exc()

    async def rotate_background(self):
        """Background image rotation for idle/sleep states"""
        while not self.initialized:
            await asyncio.sleep(0.1)
        
        print("Background rotation task started")
        
        while True:
            try:
                current_time = time.time()
                
                if self.current_state in ['idle', 'sleep', 'pout', 'boot'] and self.last_image_change is not None:
                    time_diff = current_time - self.last_image_change
                    
                    # Different rotation intervals for different states
                    rotation_interval = 0.4 if self.current_state == 'boot' else 30
                    
                    if time_diff >= rotation_interval:
                        available_images = self.image_cache.get(self.current_state, [])
                        if len(available_images) > 1:
                            current_options = [img for img in available_images if img != self.current_image]
                            if current_options:
                                new_image = random.choice(current_options)
                                self.current_image = new_image
                                self.screen.blit(self.current_image, (0, 0))
                                pygame.display.flip()
                                self.last_image_change = current_time
                
            except Exception as e:
                print(f"Error in rotate_background: {e}")
        
            # Check more frequently during boot animation
            check_interval = 0.2 if self.current_state == 'boot' else 5
            await asyncio.sleep(check_interval)
            
    def set_display_profile(self, profile: str):
        """Switch between 'normal' (LAURA) and 'claude_code' display profiles"""
        if profile not in ['normal', 'claude_code']:
            print(f"[DisplayManager] Invalid profile: {profile}, keeping current: {self.display_profile}")
            return
            
        if profile == self.display_profile:
            print(f"[DisplayManager] Already in {profile} profile")
            return
            
        print(f"[DisplayManager] Switching from {self.display_profile} to {profile} profile")
        self.display_profile = profile
        
        # Switch base path and mood set
        if profile == 'claude_code':
            self.base_path = self.claude_code_path
            self.moods = self.claude_code_moods
        else:
            self.base_path = Path('/home/carson/rp_client/assets/images/laura rp client images')
            self.moods = self.laura_moods
            
        # Update state paths
        self.states = {
            'listening': str(self.base_path / 'listening'),
            'idle': str(self.base_path / 'idle'),
            'sleep': str(self.base_path / 'sleep'),
            'pout': str(self.base_path / 'pout'),  # Pouting/hiding state
            'speaking': str(self.base_path / 'speaking'),
            'thinking': str(self.base_path / 'thinking'),
            'execution': str(self.base_path / 'execution'),
            'wake': str(self.base_path / 'wake'),
            'boot': str(self.base_path / 'boot'),
            'system': str(self.base_path / 'system'),
            'tool_use': str(self.base_path / 'tool_use'),
            'notification': str(self.base_path / 'speaking'),
            'code': str(self.base_path / 'code'),
            'error': str(self.base_path / 'error'),
            'disconnected': str(self.base_path / 'disconnected'),
        }
        
        # Clear image cache to force reload from new paths
        self.image_cache.clear()
        
        # Reload images from new directory
        self.load_image_directories()
        
        # Recalculate and resize window for new profile
        sample_image_path = self._get_sample_image_path(profile)
        new_window_size = self._calculate_adaptive_window_size(sample_image_path)
        
        if new_window_size != self.screen.get_size():
            print(f"[DisplayManager] Resizing window from {self.screen.get_size()} to {new_window_size}")
            # Ensure window stays at top when resizing
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
            self.screen = pygame.display.set_mode(new_window_size)
            pygame.display.set_caption("LAURA" if profile == 'normal' else "Claude Code")
        
        # Don't update display here - let the caller handle state management
        # The profile switch itself doesn't need to change the display state
            
    def _get_sample_image_path(self, profile):
        """Get a sample image path for window sizing"""
        if profile == 'claude_code':
            base_path = self.claude_code_path
        else:
            base_path = Path('/home/carson/rp_client/assets/images/laura rp client images')
            
        # Try idle state first, then any available state
        for state in ['idle', 'listening', 'speaking', 'thinking']:
            state_path = base_path / state
            if state_path.exists():
                png_files = list(state_path.glob('*.png'))
                if png_files:
                    return str(png_files[0])
        return None
        
    def _calculate_adaptive_window_size(self, image_path):
        """Calculate window size with 480px width, adaptive height"""
        if not image_path:
            return (480, 480)  # Fallback to square
            
        try:
            # Load image to get dimensions
            img = pygame.image.load(image_path)
            original_width, original_height = img.get_size()
            
            # Calculate height maintaining aspect ratio with 480px width
            aspect_ratio = original_height / original_width
            adaptive_height = int(480 * aspect_ratio)
            
            return (480, adaptive_height)
        except Exception as e:
            print(f"[DisplayManager] Could not load sample image {image_path}: {e}")
            return (480, 480)  # Fallback to square

    def cleanup(self):
        """Clean up pygame resources"""
        pygame.quit()

    def update_display_sync(self, state: str, mood: str = None, text: str = None):
        """Synchronous entry point for updating display from state machine callbacks.

        This method is called by the state machine when state changes occur.
        It updates the internal tracking variables and triggers async display update.
        """
        # Update tracking variables for internal use (image rotation, etc.)
        self.last_state = self.current_state
        self.current_state = state
        if mood is not None:
            self.current_mood = mood

        # Trigger async display update
        asyncio.create_task(self.update_display(state, mood, text))
