"""
Modular configuration manager for Reachy Mini client.
Handles loading, saving, and reloading of separate config files.
"""
import json
from pathlib import Path
from typing import Dict, Any
import copy

class ConfigManager:
    """Manages modular configuration files for STT, TTS, Inference, and Movement."""

    def __init__(self, config_dir: Path = None):
        """Initialize config manager with directory path."""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config_files"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Config file paths
        self.config_files = {
            "stt": self.config_dir / "stt_config.json",
            "tts": self.config_dir / "tts_config.json",
            "inference": self.config_dir / "inference_config.json",
            "movement": self.config_dir / "movement_config.json",
        }

        # Loaded configs (cache)
        self.configs: Dict[str, Dict[str, Any]] = {}

        # Load all configs on init
        self.load_all_configs()

    def load_all_configs(self):
        """Load all config files into memory."""
        for config_name, config_path in self.config_files.items():
            self.configs[config_name] = self._load_config(config_path)

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load a single config file."""
        if not config_path.exists():
            print(f"[CONFIG WARN] {config_path} not found, using empty config")
            return {}

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"[CONFIG] Loaded {config_path.name}")
            return config
        except json.JSONDecodeError as e:
            print(f"[CONFIG ERROR] Failed to decode {config_path}: {e}")
            return {}
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to load {config_path}: {e}")
            return {}

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a config by name (stt, tts, inference, movement)."""
        if config_name not in self.configs:
            print(f"[CONFIG ERROR] Unknown config: {config_name}")
            return {}
        return copy.deepcopy(self.configs[config_name])

    def update_config(self, config_name: str, new_config: Dict[str, Any]) -> bool:
        """Update a config in memory and save to file."""
        if config_name not in self.config_files:
            print(f"[CONFIG ERROR] Unknown config: {config_name}")
            return False

        try:
            # Update in-memory cache
            self.configs[config_name] = new_config

            # Save to file
            config_path = self.config_files[config_name]
            with open(config_path, 'w') as f:
                json.dump(new_config, f, indent=4)

            print(f"[CONFIG] Updated and saved {config_path.name}")
            return True
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to update {config_name}: {e}")
            return False

    def reload_config(self, config_name: str) -> bool:
        """Reload a specific config from disk."""
        if config_name not in self.config_files:
            print(f"[CONFIG ERROR] Unknown config: {config_name}")
            return False

        config_path = self.config_files[config_name]
        self.configs[config_name] = self._load_config(config_path)
        print(f"[CONFIG] Reloaded {config_name}")
        return True

    def reload_all_configs(self):
        """Reload all configs from disk."""
        print("[CONFIG] Reloading all configs...")
        self.load_all_configs()
        print("[CONFIG] All configs reloaded")

    def get_active_stt_provider(self) -> str:
        """Get the active STT provider name."""
        return self.configs.get("stt", {}).get("active_provider", "elevenlabs_scribe")

    def get_active_tts_provider(self) -> str:
        """Get the active TTS provider name."""
        return self.configs.get("tts", {}).get("active_provider", "elevenlabs")

    def get_active_inference_backend(self) -> str:
        """Get the active inference backend (mcp_server or messages_api)."""
        return self.configs.get("inference", {}).get("active_backend", "mcp_server")

    def get_stt_provider_config(self, provider: str = None) -> Dict[str, Any]:
        """Get STT provider config (defaults to active provider)."""
        if provider is None:
            provider = self.get_active_stt_provider()

        stt_config = self.configs.get("stt", {})
        return stt_config.get(provider, {})

    def get_tts_provider_config(self, provider: str = None) -> Dict[str, Any]:
        """Get TTS provider config (defaults to active provider)."""
        if provider is None:
            provider = self.get_active_tts_provider()

        tts_config = self.configs.get("tts", {})
        return tts_config.get(provider, {})

    def get_inference_backend_config(self, backend: str = None) -> Dict[str, Any]:
        """Get inference backend config (defaults to active backend)."""
        if backend is None:
            backend = self.get_active_inference_backend()

        inference_config = self.configs.get("inference", {})
        return inference_config.get(backend, {})


# Global instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
