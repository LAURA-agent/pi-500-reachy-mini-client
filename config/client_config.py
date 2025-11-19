# --- START OF client_config_v4.2.py (with Vosk Model Fix) ---

import json
from pathlib import Path
import copy

# ========== Paths ==========
BASE_PATH = Path("/home/user/reachy/pi_reachy_deployment")
SOUND_BASE_PATH = BASE_PATH / "assets/sounds"

DISPLAY_SVG_PATH_DEFAULT = BASE_PATH / "svg files/silhouette.svg"
DISPLAY_BOOT_IMG_PATH_DEFAULT = BASE_PATH / "assets/images/CC_images/boot/boot01.png"
DISPLAY_WINDOW_SIZE_DEFAULT = 512

QUERY_FILES_DIR_DEFAULT = BASE_PATH / "query_files"
QUERY_OFFLOAD_DIR_DEFAULT = BASE_PATH / "query_offload"


# ========== Wakeword Configuration ==========
WAKEWORD_MODEL_DIR_DEFAULT = BASE_PATH / "snowboy"
WAKEWORD_RESOURCE_FILE_DEFAULT = BASE_PATH / "snowboy/resources/common.res"
WAKE_WORDS_AND_SENSITIVITIES_DEFAULT = {
    "GD_Laura.pmdl": 0.35,          # Lower = easier to trigger
    "Wake_up_Laura.pmdl": 0.35,     # Lower = easier to trigger
    "Laura.pmdl": 0.35,             # Lower = easier to trigger (was 0.65 - too hard!)
    "tookmycrazypills.pmdl": 0.35,  # Lower = easier to trigger
    "claudecode.pmdl": 0.35,        # Lower = easier to trigger
    "sendnote.pmdl": 0.35,          # Lower = easier to trigger
}

# ========== Audio Settings ==========
AUDIO_SAMPLE_RATE_DEFAULT = 16000
SNOWBOY_AUDIO_CHUNK_SIZE_DEFAULT = 2048

# ========== Default TTS Provider Settings (Global Fallbacks) ==========
DEFAULT_ELEVENLABS_VOICE_NAME = "qEwI395unGwWV1dn3Y65"
DEFAULT_ELEVENLABS_MODEL = "eleven_flash_v2_5"
DEFAULT_CARTESIA_VOICE_ID = "78f71eb3-187f-48b4-a763-952f2f4f838a"
DEFAULT_CARTESIA_MODEL = "sonic-en"
DEFAULT_PIPER_MODEL_PATH = str(BASE_PATH / "piper_models/en_US-ljspeech-low.onnx")
DEFAULT_PIPER_VOICE_NAME = "ljspeech"

# ========== STT Settings ==========
# --- MODIFIED: Define Vosk Model Paths ---
VOSK_MODELS_BASE_PATH = BASE_PATH / "models"
VOSK_MODEL_PATHS_AVAILABLE = {
    "small": str(VOSK_MODELS_BASE_PATH / "vosk-model-small-en-us-0.15"),
    "medium": str(VOSK_MODELS_BASE_PATH / "vosk-model-en-us-0.22"),
    # Add other Vosk models here if you have them, e.g., "en-in" for Indian English
}
VOSK_MODEL_SIZE_DEFAULT = "small" # Set your preferred default size here


# ========== Device/Server & Misc ==========
DEVICE_ID_DEFAULT = "reachy_mini_pi500"

# Network Configuration:
# - Maestas Mansion WiFi (LAN): 192.168.0.128:8765 (Mac Mini local)
# - External/Mobile (Public): 174.165.47.128:8765 (Mac Mini public)
SERVER_URL_DEFAULT = "http://174.165.47.128:8765"  # Default to public IP
SERVER_URL_LAN = "http://192.168.0.128:8765"        # LAN IP for Maestas Mansion
MAESTAS_MANSION_WIFI_SSID = "Maestas Mansion"       # WiFi network name to detect

KEEP_TEMP_AUDIO_FILES_DEFAULT = False

# ========== Mood Color Configuration ==========
MOOD_COLORS = {
    "annoyed": {"name": "annoyed", "gradient_colors": [(255, 99, 71), (255, 160, 122)]},
    "caring": {"name": "caring", "gradient_colors": [(173, 216, 230), (255, 255, 255)]},
    "casual": {"name": "casual", "gradient_colors": [(255, 105, 180), (0, 206, 209)]},
    "cheerful": {"name": "cheerful", "gradient_colors": [(255, 255, 0), (255, 140, 0)]},
    "concerned": {"name": "concerned", "gradient_colors": [(255, 165, 0), (255, 215, 0)]},
    "confused": {"name": "confused", "gradient_colors": [(128, 0, 128), (218, 112, 214)]},
    "curious": {"name": "curious", "gradient_colors": [(0, 191, 255), (135, 206, 250)]},
    "disappointed": {"name": "disappointed", "gradient_colors": [(105, 105, 105), (169, 169, 169)]},
    "embarrassed": {"name": "embarrassed", "gradient_colors": [(255, 192, 203), (255, 99, 71)]},
    "sassy": {"name": "sassy", "gradient_colors": [(255, 69, 0), (255, 165, 0)]},
    "scared": {"name": "scared", "gradient_colors": [(0, 0, 0), (105, 105, 105)]},
    "surprised": {"name": "surprised", "gradient_colors": [(147, 112, 219), (216, 191, 216)]},
    "suspicious": {"name": "suspicious", "gradient_colors": [(85, 107, 47), (124, 252, 0)]},
    "thoughtful": {"name": "thoughtful", "gradient_colors": [(70, 130, 180), (100, 149, 237)]},

    "booting": {"name": "booting", "gradient_colors": [(255, 255, 255), (192, 192, 192)]},
    "error": {"name": "error", "gradient_colors": [(255, 0, 0), (178, 34, 34)]},
    "disconnected": {"name": "disconnected", "gradient_colors": [(50, 50, 50), (100, 100, 100)]},

    "default": {"name": "casual", "gradient_colors": [(255, 105, 180), (0, 206, 209)]}
}


# ========== Client Settings Management (Loaded at runtime) ==========
_DEFAULT_CLIENT_SETTINGS = {
    "enable_display_audio_visualizer": True,    
    "tts_mode": "api",
    "api_tts_provider": "elevenlabs",
    "stt_mode": "local",
    "vosk_model_size": VOSK_MODEL_SIZE_DEFAULT, # --- NEW: Vosk model size setting ---

    "SERVER_URL": SERVER_URL_DEFAULT,
    "DEVICE_ID": DEVICE_ID_DEFAULT,
    "KEEP_TEMP_AUDIO_FILES": KEEP_TEMP_AUDIO_FILES_DEFAULT,
    "AUDIO_SAMPLE_RATE": AUDIO_SAMPLE_RATE_DEFAULT,
    "SNOWBOY_AUDIO_CHUNK_SIZE": SNOWBOY_AUDIO_CHUNK_SIZE_DEFAULT,
    "WAKEWORD_MODEL_DIR": str(WAKEWORD_MODEL_DIR_DEFAULT),
    "WAKEWORD_RESOURCE_FILE": str(WAKEWORD_RESOURCE_FILE_DEFAULT),
    "WAKE_WORDS_AND_SENSITIVITIES": WAKE_WORDS_AND_SENSITIVITIES_DEFAULT,

    "DISPLAY_SVG_PATH": str(DISPLAY_SVG_PATH_DEFAULT),
    "DISPLAY_BOOT_IMG_PATH": str(DISPLAY_BOOT_IMG_PATH_DEFAULT),
    "DISPLAY_WINDOW_SIZE": DISPLAY_WINDOW_SIZE_DEFAULT,

    "QUERY_FILES_DIR": str(QUERY_FILES_DIR_DEFAULT),
    "QUERY_OFFLOAD_DIR": str(QUERY_OFFLOAD_DIR_DEFAULT),

    "elevenlabs_default_voice_name": DEFAULT_ELEVENLABS_VOICE_NAME,
    "elevenlabs_default_model": DEFAULT_ELEVENLABS_MODEL,
    "cartesia_default_voice_id": DEFAULT_CARTESIA_VOICE_ID,
    "cartesia_default_model": DEFAULT_CARTESIA_MODEL,
    "piper_default_model_path": DEFAULT_PIPER_MODEL_PATH,
    "piper_default_voice_name": DEFAULT_PIPER_VOICE_NAME,



    "persona_voice_configs": {
        "laura": {
            "elevenlabs": {"voice_name_or_id": "qEwI395unGwWV1dn3Y65", "model": "eleven_flash_v2_5"},
            "cartesia": {"voice_id": "78f71eb3-187f-48b4-a763-952f2f4f838a", "model": "sonic-en"},
            "piper": {"model_path": DEFAULT_PIPER_MODEL_PATH, "voice_name": DEFAULT_PIPER_VOICE_NAME}
        },
        "max": {
            "elevenlabs": {"voice_name_or_id": "uY96J30mUhYUIymmD5cu", "model": "eleven_flash_v2_5"},
            "cartesia": {"voice_id": "another-cartesia-id-for-max", "model": "sonic-en"},
            "piper": {"model_path": "/path/to/some/other_piper_model.onnx", "voice_name": "speaker_for_max"}
        },
        "client_default": {
            "elevenlabs": {"voice_name_or_id": "qEwI395unGwWV1dn3Y65", "model": "eleven_flash_v2_5"},
            "cartesia": {"voice_id": DEFAULT_CARTESIA_VOICE_ID, "model": DEFAULT_CARTESIA_MODEL},
            "piper": {"model_path": DEFAULT_PIPER_MODEL_PATH, "voice_name": DEFAULT_PIPER_VOICE_NAME}
        }
    },
    "vad_settings": {
        "silero_threshold": 0.5,
        "silence_duration": 2.0,
        "min_speech_duration": 0.4,
        "speech_buffer_time": 1.0,
        "max_recording_time": 45.0,
        "chunk_size": 2048
    },
    "_default_config": {}
}
_DEFAULT_CLIENT_SETTINGS["_default_config"] = copy.deepcopy(_DEFAULT_CLIENT_SETTINGS)

CLIENT_CONFIG_PATH = BASE_PATH / "client_settings.json"
client_settings = {}

def load_client_settings():
    global client_settings
    settings = copy.deepcopy(_DEFAULT_CLIENT_SETTINGS)
    
    if CLIENT_CONFIG_PATH.exists():
        try:
            with open(CLIENT_CONFIG_PATH, "r") as f:
                user_config = json.load(f)
            for key, value in user_config.items():
                if isinstance(settings.get(key), dict) and isinstance(value, dict) and key not in ["_default_config"]:
                    settings[key].update(value)
                elif key not in ["_default_config"]:
                    settings[key] = value
        except json.JSONDecodeError:
            print(f"[CONFIG ERROR] Failed to decode {CLIENT_CONFIG_PATH}. Using defaults.")
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to load {CLIENT_CONFIG_PATH}: {e}. Using defaults.")
    else:
        print(f"[CONFIG INFO] {CLIENT_CONFIG_PATH} not found. Using default settings and creating file.")
        save_client_settings(settings)

    client_settings.clear()
    client_settings.update(settings)
    
    # Ensure VAD_SETTINGS global in this module is set from loaded settings
    global VAD_SETTINGS
    VAD_SETTINGS = client_settings.get("vad_settings", _DEFAULT_CLIENT_SETTINGS["vad_settings"])

    # Ensure other global variables in this module are set from loaded settings for client script imports
    global SERVER_URL, DEVICE_ID, KEEP_TEMP_AUDIO_FILES, AUDIO_SAMPLE_RATE, SNOWBOY_AUDIO_CHUNK_SIZE
    global WAKEWORD_MODEL_DIR, WAKEWORD_RESOURCE_FILE, WAKE_WORDS_AND_SENSITIVITIES
    global VOSK_MODEL_PATH # --- MODIFIED ---
    
    SERVER_URL = client_settings.get("SERVER_URL", SERVER_URL_DEFAULT)
    DEVICE_ID = client_settings.get("DEVICE_ID", DEVICE_ID_DEFAULT)
    KEEP_TEMP_AUDIO_FILES = client_settings.get("KEEP_TEMP_AUDIO_FILES", KEEP_TEMP_AUDIO_FILES_DEFAULT)
    AUDIO_SAMPLE_RATE = client_settings.get("AUDIO_SAMPLE_RATE", AUDIO_SAMPLE_RATE_DEFAULT)
    SNOWBOY_AUDIO_CHUNK_SIZE = client_settings.get("SNOWBOY_AUDIO_CHUNK_SIZE", SNOWBOY_AUDIO_CHUNK_SIZE_DEFAULT)
    WAKEWORD_MODEL_DIR = Path(client_settings.get("WAKEWORD_MODEL_DIR", WAKEWORD_MODEL_DIR_DEFAULT))
    WAKEWORD_RESOURCE_FILE = Path(client_settings.get("WAKEWORD_RESOURCE_FILE", WAKEWORD_RESOURCE_FILE_DEFAULT))
    WAKE_WORDS_AND_SENSITIVITIES = client_settings.get("WAKE_WORDS_AND_SENSITIVITIES", WAKE_WORDS_AND_SENSITIVITIES_DEFAULT)
    
    # --- MODIFIED: VOSK_MODEL_PATH is now derived from VOSK_MODEL_SIZE ---
    VOSK_MODEL_PATH = get_vosk_model_path() 

    return client_settings

def save_client_settings(current_config_dict=None):
    if current_config_dict is None:
        current_config_dict = client_settings
    config_to_save = {k: v for k, v in current_config_dict.items() if k != "_default_config"}
    try:
        CLIENT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CLIENT_CONFIG_PATH, "w") as f:
            json.dump(config_to_save, f, indent=4)
        print(f"[CONFIG INFO] Client settings saved to {CLIENT_CONFIG_PATH}")
    except Exception as e:
        print(f"[CONFIG ERROR] Failed to save client settings: {e}")

# --- Helper functions for runtime access ---

def get_active_tts_provider() -> str:
    return client_settings.get("api_tts_provider", _DEFAULT_CLIENT_SETTINGS["api_tts_provider"])

def set_active_tts_provider(new_provider: str):
    if new_provider in ["cartesia", "elevenlabs", "piper"]:
        client_settings["api_tts_provider"] = new_provider
        save_client_settings()
        print(f"[CONFIG UPDATE] API TTS Provider set to: {new_provider}")
    else:
        print(f"[CONFIG WARN] Invalid API TTS Provider: {new_provider}. No change made.")

def get_voice_params_for_persona(provider_name: str, persona_name: str | None) -> dict:
    settings = client_settings
    
    effective_persona_name = persona_name if persona_name else "client_default"
    
    persona_configs_all = settings.get("persona_voice_configs", {})
    
    persona_provider_config = persona_configs_all.get(effective_persona_name, {}).get(provider_name)

    if not persona_provider_config and effective_persona_name != "client_default":
        persona_provider_config = persona_configs_all.get("client_default", {}).get(provider_name)

    final_config = {}
    if provider_name == "elevenlabs":
        default_voice = settings.get("elevenlabs_default_voice_name")
        default_model = settings.get("elevenlabs_default_model")
        final_config["voice_name_or_id"] = (persona_provider_config or {}).get("voice_name_or_id", default_voice)
        final_config["model"] = (persona_provider_config or {}).get("model", default_model)
    elif provider_name == "cartesia":
        default_id = settings.get("cartesia_default_voice_id")
        default_model = settings.get("cartesia_default_model")
        final_config["voice_id"] = (persona_provider_config or {}).get("voice_id", default_id)
        final_config["model"] = (persona_provider_config or {}).get("model", default_model)
    elif provider_name == "piper":
        default_path = settings.get("piper_default_model_path")
        default_voice = settings.get("piper_default_voice_name")
        final_config["model_path"] = (persona_provider_config or {}).get("model_path", default_path)
        final_config["voice_name"] = (persona_provider_config or {}).get("voice_name", default_voice)
    else:
        return {}

    if not any(final_config.values()):
        return {}
        
    return final_config

def get_mood_color_config(mood_name_str: str) -> dict:
    """
    Returns the color and name configuration for a given mood string.
    Falls back to 'default' if mood not found.
    """
    return MOOD_COLORS.get(str(mood_name_str).lower(), MOOD_COLORS["default"])

def get_vosk_model_path() -> Path:
    """
    Returns the Path to the selected Vosk model directory based on VOSK_MODEL_SIZE.
    """
    model_size = client_settings.get("vosk_model_size", VOSK_MODEL_SIZE_DEFAULT)
    path_str = VOSK_MODEL_PATHS_AVAILABLE.get(model_size)
    
    if path_str:
        model_path = Path(path_str)
        if model_path.exists():
            return model_path
        else:
            print(f"[CONFIG ERROR] Vosk model '{model_size}' path does not exist: {model_path}. Falling back to default small model.")
            # Fallback to small if requested size not found on disk
            fallback_path_str = VOSK_MODEL_PATHS_AVAILABLE.get(VOSK_MODEL_SIZE_DEFAULT)
            if fallback_path_str:
                fallback_path = Path(fallback_path_str)
                if fallback_path.exists():
                    return fallback_path
                else:
                    print(f"[CONFIG WARNING] Default Vosk model '{VOSK_MODEL_SIZE_DEFAULT}' path also does not exist: {fallback_path}.")
                    print("[CONFIG INFO] Vosk not available - speech features will be disabled. Use Gradio interface for input.")
                    return None  # Return None instead of raising error
            else:
                raise ValueError("VOSK_MODEL_SIZE_DEFAULT not found in VOSK_MODEL_PATHS_AVAILABLE. Config error.")
    else:
        print(f"[CONFIG ERROR] Vosk model size '{model_size}' not defined in VOSK_MODEL_PATHS_AVAILABLE. Using default small model.")
        # Fallback to small if size not defined in map
        fallback_path_str = VOSK_MODEL_PATHS_AVAILABLE.get(VOSK_MODEL_SIZE_DEFAULT)
        if fallback_path_str:
            fallback_path = Path(fallback_path_str)
            if fallback_path.exists():
                return fallback_path
            else:
                print(f"[CONFIG WARNING] Vosk model size '{model_size}' not defined, and default '{VOSK_MODEL_SIZE_DEFAULT}' also not found at {fallback_path}.")
                print("[CONFIG INFO] Vosk not available - speech features will be disabled. Use Gradio interface for input.")
                return None  # Return None instead of raising error
        else:
            raise ValueError("VOSK_MODEL_SIZE_DEFAULT not found in VOSK_MODEL_PATHS_AVAILABLE. Config error.")

def get_network_appropriate_server_url() -> str:
    """
    Detect current WiFi network and return appropriate server URL.
    Returns LAN IP if connected to Maestas Mansion, otherwise public IP.
    """
    try:
        import subprocess
        # Get current WiFi SSID on Linux
        result = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            current_ssid = result.stdout.strip()
            if current_ssid == MAESTAS_MANSION_WIFI_SSID:
                print(f"[CONFIG INFO] Connected to {MAESTAS_MANSION_WIFI_SSID}, using LAN server: {SERVER_URL_LAN}")
                return SERVER_URL_LAN
        
        # Fallback: check if we can reach the LAN server (ping test)
        lan_host = SERVER_URL_LAN.replace('http://', '').split(':')[0]  # Extract IP
        ping_result = subprocess.run(['ping', '-c', '1', '-W', '2', lan_host], 
                                   capture_output=True, timeout=5)
        if ping_result.returncode == 0:
            print(f"[CONFIG INFO] LAN server reachable, using: {SERVER_URL_LAN}")
            return SERVER_URL_LAN
            
    except Exception as e:
        print(f"[CONFIG WARN] Network detection failed: {e}")
    
    # Default to public IP
    print(f"[CONFIG INFO] Using default server: {SERVER_URL_DEFAULT}")
    return SERVER_URL_DEFAULT


# --- Initialize settings and global variables on module import ---
# These global variables are directly imported by pi500_mcp_client.py
# and should reflect the loaded client_settings.
# They are declared and then populated by load_client_settings().
SERVER_URL = None
DEVICE_ID = None
KEEP_TEMP_AUDIO_FILES = None
VAD_SETTINGS = {} 
AUDIO_SAMPLE_RATE = None
SNOWBOY_AUDIO_CHUNK_SIZE = None
WAKEWORD_MODEL_DIR = None
WAKEWORD_RESOURCE_FILE = None
WAKE_WORDS_AND_SENSITIVITIES = None
VOSK_MODEL_PATH = None # This will now be set by get_vosk_model_path()

load_client_settings() # Call this to load settings on module import
