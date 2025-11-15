# Scene Manager Architecture Problems

## Current BROKEN Flow:
1. Director generates static scene setup ONCE at beginning
2. Orchestrator passes same directions to agents for ALL turns
3. Agents generate responses with static directions
4. AFTER generation, Director might "intervene" to override speaker choice
5. Director's actual instructions never reach the agents

## The Fundamental Flaw:
**The Director is an afterthought, not the conductor**

### What Actually Happens:
```
Turn 1: Laura generates response -> picks Claude -> Director does nothing
Turn 2: Claude generates response -> picks Laura -> Director does nothing
Turn 3+: Agent generates -> Director MIGHT override speaker AFTER the fact
```

### Problems:
- Director sees responses AFTER they're generated (too late to shape them)
- Director can only override speaker choice, not influence content
- "Private directions" are static strings, not dynamic guidance
- Director's "interventions" are logged but don't affect anything
- Whole system is agent-driven, not director-driven

## What SHOULD Happen:

### Director-First Architecture:
```
BEFORE each turn:
1. Director analyzes scene state
2. Director chooses who speaks next
3. Director provides specific instructions for that character
4. Character generates response WITH those instructions
5. Response goes to output
```

### Director Powers Needed:
- **Pre-turn decision**: Choose speaker BEFORE generation
- **Dynamic direction**: Give specific instructions each turn
- **Content shaping**: "Say X but imply Y"
- **Emotional guidance**: "Be defensive here"
- **Plot injection**: "Reveal that you knew all along"

## The Cache Problem Is A Symptom:

The reason cache is broken is because we're trying to cache "scene-specific" directions that should be DYNAMIC per turn. The architecture assumes static directions when it should be:

1. **Static (cached)**: Character personality, basic rules
2. **Dynamic per scene**: Overall topic, setting
3. **Dynamic per turn**: Director's specific instructions

## File Organization Disaster:

```
scene_manager/
├── enhanced_* files (current attempt)
├── archive/ (50+ old attempts)
├── agents/ (character definitions mixed with logic)
├── gameboy_devices/ (should be separate)
├── Random test files
├── Multiple orchestrators
└── No clear entry point
```

## The Solution:

**Start over with Director-first architecture**
- Director is THE brain
- Agents are just actors following direction
- Clear separation of concerns
- Visible decision logging
- Clean file structure