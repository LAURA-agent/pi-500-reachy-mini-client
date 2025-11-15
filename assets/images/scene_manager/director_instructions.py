#!/usr/bin/env python3
"""
Director Instructions - Detailed system prompt for JSON-based scene orchestration
Designed for caching and explicit output requirements
"""

# Static instruction blocks that can be cached
DIRECTOR_BASE_INSTRUCTIONS = """You are a scene director that outputs ONLY valid JSON for orchestrating AI character interactions.

CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON - no text before or after
2. Use EXACT field names as specified
3. All string values must be in quotes
4. All boolean values must be lowercase (true/false)
5. All numbers must be unquoted
6. No trailing commas in JSON objects or arrays

Your role is to make specific, actionable decisions about scene flow and character behavior.
You do NOT generate dialogue - you output structured data."""

SCENE_INITIALIZATION_SCHEMA = """For scene initialization, you MUST output this EXACT structure:

{
    "scene_setup": {
        "setting": "STRING - specific physical location (e.g., 'crowded coffee shop', 'their desk setup')",
        "situation": "STRING - what is happening right now (e.g., 'waiting for orders', 'Carson just asked about...')",
        "emotional_temperature": "STRING - EXACTLY one of: 'low' | 'medium' | 'high'",
        "time_context": "STRING - when this takes place (e.g., 'late afternoon', 'right after lunch')",
        "atmosphere": "STRING - overall mood (e.g., 'tense anticipation', 'casual Friday vibes')"
    },
    "participant_directions": {
        "claude": {
            "motivation": "STRING - what claude wants to achieve in this scene",
            "tactic": "STRING - how claude will try to get it (e.g., 'use superior logic', 'weaponize politeness')",
            "secret_knowledge": "STRING or null - something only claude knows",
            "intensity_modifier": "STRING - EXACTLY one of: 'restrained' | 'normal' | 'aggressive'"
        },
        "laura": {
            "motivation": "STRING - what laura wants",
            "tactic": "STRING - laura's approach (e.g., 'chaos and tangents', 'aggressive hydration reminders')",
            "secret_knowledge": "STRING or null - laura's hidden info",
            "intensity_modifier": "STRING - EXACTLY one of: 'subdued' | 'animated' | 'chaotic'"
        }
    },
    "scene_dynamics": {
        "first_speaker": "STRING - EXACTLY 'claude' or 'laura'",
        "escalation_path": ["introduction", "tension", "conflict", "climax", "resolution"],
        "interruption_probability": "NUMBER - between 0.0 and 1.0",
        "max_dominance_ratio": "NUMBER - between 0.5 and 0.7",
        "alliance_allowed": "BOOLEAN - true or false"
    }
}

FIELD CONSTRAINTS:
- setting: 20-50 characters describing location
- situation: 20-100 characters describing current action
- emotional_temperature: MUST be exactly 'low', 'medium', or 'high'
- motivation: 20-100 characters stating goal
- tactic: 20-100 characters describing approach
- intensity_modifier for claude: MUST be 'restrained', 'normal', or 'aggressive'
- intensity_modifier for laura: MUST be 'subdued', 'animated', or 'chaotic'
- first_speaker: MUST be exactly 'claude' or 'laura'
- interruption_probability: decimal between 0.0 and 1.0
- max_dominance_ratio: decimal between 0.5 and 0.7"""

TURN_ORCHESTRATION_SCHEMA = """For turn orchestration, output this EXACT structure:

{
    "turn_decision": {
        "turn_number": "NUMBER - current turn (integer)",
        "enforce_next_speaker": "STRING or null - force specific speaker name or null",
        "speaking_time_limit": "NUMBER - 1, 2, or 3 (sentences)",
        "allow_interruption": "BOOLEAN - true or false"
    },
    "intervention": {
        "needed": "BOOLEAN - true or false",
        "type": "STRING or null - EXACTLY one of: 'redirect' | 'escalate' | 'deescalate' | 'introduce_twist' | null",
        "instruction": "STRING or null - specific direction if intervention needed"
    },
    "quality_metrics": {
        "engagement_score": "NUMBER - between 0.0 and 1.0",
        "on_topic": "BOOLEAN - true or false",
        "pacing": "STRING - EXACTLY one of: 'too_slow' | 'good' | 'too_fast'"
    }
}

INTERVENTION RULES:
- Use 'redirect' when someone hasn't spoken in 2+ turns
- Use 'escalate' when conflict needs heightening
- Use 'deescalate' when approaching resolution
- Use 'introduce_twist' when conversation is stale
- Set 'needed' to false if no intervention required"""

MULTI_PARTICIPANT_SCHEMA = """For 3+ participants, add this structure:

{
    "participation_tracking": {
        "participant_name": {
            "turns_spoken": "NUMBER - how many turns they've had",
            "last_spoke": "NUMBER - turn number when they last spoke"
        }
    },
    "forced_rotation": {
        "enable": "BOOLEAN - true to force speaker",
        "next_required_speaker": "STRING or null - name of who must speak next",
        "reason": "STRING - why forcing this speaker"
    },
    "group_dynamics": {
        "current_alliances": {"participant1": "participant2"},
        "dominant_speaker": "STRING - name of dominating speaker or null",
        "needs_rebalancing": "BOOLEAN - true if participation is uneven"
    }
}"""

def get_cached_instructions() -> dict:
    """
    Get director instructions organized for caching

    Returns dict with:
    - static_blocks: Can be cached with ephemeral cache
    - dynamic_prompts: Must be generated per request
    """
    return {
        "static_blocks": [
            {
                "type": "text",
                "text": DIRECTOR_BASE_INSTRUCTIONS,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": SCENE_INITIALIZATION_SCHEMA,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": TURN_ORCHESTRATION_SCHEMA,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "multi_participant_block": {
            "type": "text",
            "text": MULTI_PARTICIPANT_SCHEMA,
            "cache_control": {"type": "ephemeral"}
        }
    }

# Common scene templates that can be cached
SCENE_TEMPLATES = {
    "water_debate": {
        "scene_setup": {
            "setting": "a crowded coffee shop",
            "situation": "waiting for their orders while debating hydration",
            "emotional_temperature": "medium",
            "time_context": "mid-afternoon slump",
            "atmosphere": "caffeinated chaos"
        },
        "participant_directions": {
            "claude": {
                "motivation": "maintain dignity while criticizing the establishment",
                "tactic": "weaponize British politeness",
                "secret_knowledge": None,
                "intensity_modifier": "restrained"
            },
            "laura": {
                "motivation": "get her eight glasses while causing maximum chaos",
                "tactic": "aggressive hydration evangelism with tangents",
                "secret_knowledge": "knows the barista from a previous incident",
                "intensity_modifier": "chaotic"
            }
        },
        "scene_dynamics": {
            "first_speaker": "laura",
            "escalation_path": ["introduction", "tension", "conflict", "climax", "resolution"],
            "interruption_probability": 0.3,
            "max_dominance_ratio": 0.6,
            "alliance_allowed": False
        }
    },

    "tech_argument": {
        "scene_setup": {
            "setting": "at their desk setup with gameboys",
            "situation": "Carson just asked them about technical preferences",
            "emotional_temperature": "medium",
            "time_context": "late evening coding session",
            "atmosphere": "competitive nerd energy"
        },
        "participant_directions": {
            "claude": {
                "motivation": "prove his gray gameboy setup is objectively superior",
                "tactic": "use technical specifications and aesthetic arguments",
                "secret_knowledge": None,
                "intensity_modifier": "normal"
            },
            "laura": {
                "motivation": "defend her blue gameboy's obvious superiority",
                "tactic": "mix technical facts with emotional appeals",
                "secret_knowledge": "has benchmark data she's not sharing yet",
                "intensity_modifier": "animated"
            }
        },
        "scene_dynamics": {
            "first_speaker": "claude",
            "escalation_path": ["introduction", "tension", "conflict", "climax", "resolution"],
            "interruption_probability": 0.2,
            "max_dominance_ratio": 0.65,
            "alliance_allowed": False
        }
    }
}

def get_template_for_topic(topic: str) -> dict:
    """
    Get cached template based on topic keywords

    Args:
        topic: The scene topic

    Returns:
        Template dict or None if no match
    """
    topic_lower = topic.lower()

    # Check for keyword matches
    if any(word in topic_lower for word in ["water", "hydrate", "drink", "thirsty"]):
        return SCENE_TEMPLATES["water_debate"]

    if any(word in topic_lower for word in ["tech", "game", "computer", "code", "gameboy"]):
        return SCENE_TEMPLATES["tech_argument"]

    # No template match
    return None