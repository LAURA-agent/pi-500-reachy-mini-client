"""Pout mode tools for LLM control of Laura's pouting behavior.

These tools allow the LLM to trigger and control pout mode, including:
- Entering pout state
- Rotating body while pouting
- Playing antenna animations
- Exiting pout (only accessible after user interaction)
"""

def enter_pout_mode(entry_speed: str = "slow") -> str:
    """Enter pouting state - Laura hunches into sleep pose and becomes upset.

    This triggers Laura to:
    - Move into hunched/hiding sleep pose
    - Display pout expression
    - Freeze breathing motion
    - Become less responsive to normal wake words

    Args:
        entry_speed: How quickly to enter pout pose. Options:
            - "instant" (0.5s): Quick comedic retreat, snappy and reactive
            - "slow" (2.0s): Deliberate controlled disdain, methodical and pointed

    Returns:
        Confirmation message
    """
    # This will be called by the LLM during conversation
    # The actual implementation is in run_pi_reachy.py via MCP server
    speed_desc = "quickly" if entry_speed == "instant" else "slowly and deliberately"
    return f"Entering pout mode {speed_desc} - Laura is now pouting and hiding"


def pout_rotate_to(angle: int) -> str:
    """Rotate Laura's body while in pout mode to face a specific direction.

    Args:
        angle: Target rotation angle in degrees. Must be one of: -90, -45, 0, 45, 90
               0 = facing forward
               -90 = turned 90° left (away from user on left)
               +90 = turned 90° right (away from user on right)

    Returns:
        Confirmation message

    Example:
        >>> pout_rotate_to(-90)
        "Laura rotates away from you, turning her back"
    """
    allowed_angles = [-90, -45, 0, 45, 90]
    if angle not in allowed_angles:
        closest = min(allowed_angles, key=lambda x: abs(x - angle))
        return f"Invalid angle {angle}°. Using closest allowed angle: {closest}°"

    direction = "away" if abs(angle) > 45 else "slightly"
    return f"Laura rotates {direction} ({angle}°) while staying in pout pose"


def play_antenna_pattern(pattern: str = "frustration_twitch") -> str:
    """Trigger antenna animation pattern while pouting.

    Args:
        pattern: Animation pattern name. Options:
            - "frustration_twitch" (default): Rapid ±3° flicks at 1-1.5s intervals (Sailor Moon style)
            - "angry_swing": Larger ±5° swings
            - "nervous_flutter": Quick small ±1.5° movements

    Returns:
        Confirmation message

    Example:
        >>> play_antenna_pattern("frustration_twitch")
        "Laura's antennas twitch in frustration"
    """
    valid_patterns = ["frustration_twitch", "angry_swing", "nervous_flutter"]
    if pattern not in valid_patterns:
        return f"Invalid pattern '{pattern}'. Valid options: {', '.join(valid_patterns)}"

    descriptions = {
        "frustration_twitch": "Laura's antennas twitch rapidly in frustration",
        "angry_swing": "Laura's antennas swing back and forth angrily",
        "nervous_flutter": "Laura's antennas flutter nervously"
    }

    return descriptions[pattern]


def exit_pout_mode(reason: str = "user_apologized") -> str:
    """Exit pout mode and return to normal state.

    IMPORTANT: This should only be called after appropriate user interaction,
    such as an apology, wake word, or other trigger defined in wake_words.yml.

    Args:
        reason: Why Laura is exiting pout mode. Options:
            - "user_apologized": User said sorry
            - "user_called_out": User told Laura to quit being a baby
            - "time_passed": Laura decided to let it go
            - "distraction": Something else caught Laura's attention

    Returns:
        Confirmation message

    Note:
        The exit animation will be selected based on the wake word that triggered
        this, or a default gentle exit if called directly by LLM.
    """
    return f"Exiting pout mode - {reason}"


# Tool metadata for MCP server registration
POUT_TOOLS = [
    {
        "name": "enter_pout_mode",
        "description": "Make Laura enter pouting state - she hunches into sleep pose, displays pout expression, and becomes upset. Use when Laura is frustrated, offended, or wants to retreat. Choose 'instant' for comedic snappy reactions or 'slow' for deliberate disdainful retreats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entry_speed": {
                    "type": "string",
                    "description": "Speed of entry: 'instant' (0.5s, comedic/reactive) or 'slow' (2.0s, deliberate/disdainful)",
                    "enum": ["instant", "slow"],
                    "default": "slow"
                }
            },
            "required": []
        }
    },
    {
        "name": "pout_rotate_to",
        "description": "Rotate Laura's body while pouting to face a specific direction. Useful for turning away from the user dramatically. Allowed angles: -90, -45, 0, 45, 90 degrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "integer",
                    "description": "Target rotation angle in degrees (-90, -45, 0, 45, 90)",
                    "enum": [-90, -45, 0, 45, 90]
                }
            },
            "required": ["angle"]
        }
    },
    {
        "name": "play_antenna_pattern",
        "description": "Trigger antenna animation while pouting to express emotions. Patterns: frustration_twitch (Sailor Moon style rapid flicks), angry_swing (larger movements), nervous_flutter (small rapid movements).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Animation pattern name",
                    "enum": ["frustration_twitch", "angry_swing", "nervous_flutter"],
                    "default": "frustration_twitch"
                }
            },
            "required": []
        }
    },
    {
        "name": "exit_pout_mode",
        "description": "Exit pout mode and return to normal state. Should only be used after user apologizes, calls Laura out, or other appropriate trigger. The exit animation depends on the wake word used.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why Laura is exiting pout mode",
                    "enum": ["user_apologized", "user_called_out", "time_passed", "distraction"],
                    "default": "user_apologized"
                }
            },
            "required": []
        }
    }
]
