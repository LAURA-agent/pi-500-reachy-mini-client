# Scene Manager - Director-First Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SCENE ORCHESTRATION SYSTEM                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  iOS App          HTTP POST :8787         Scene Server           │
│  ┌──────┐         ┌──────────────┐       (scene_server.py)      │
│  │ User │────────>│  /scene/start │       ┌──────────────┐       │
│  └──────┘         └──────────────┘        │   FastAPI    │       │
│                                            │   Port 8787  │       │
│                                            └──────┬───────┘       │
│                                                   │               │
│                          ┌────────────────────────▼──────┐        │
│                          │   SCENE ORCHESTRATOR          │        │
│                          │  (orchestrator.py)            │        │
│                          │  - Manages turn flow          │        │
│                          │  - Coordinates all components │        │
│                          └────────┬──────────────────────┘        │
│                                   │                               │
│        ┌──────────────────────────▼──────────────────────┐       │
│        │          SCENE DIRECTOR AGENT                   │       │
│        │       (scene_director_agent.py)                 │       │
│        │                                                  │       │
│        │  THE BRAIN - Makes ALL Decisions:               │       │
│        │  • Who speaks next                              │       │
│        │  • What instruction they get                    │       │
│        │  • Emotional tone                               │       │
│        │  • Scene progression                            │       │
│        └────────────────────┬─────────────────────────────┘       │
│                             │                                     │
│                  Director's Instructions                          │
│                             │                                     │
│         ┌───────────────────▼───────────────────┐                │
│         │         CHARACTER AGENTS              │                │
│         │   ┌─────────────┐ ┌─────────────┐    │                │
│         │   │claude_agent │ │laura_agent  │    │                │
│         │   │             │ │             │    │                │
│         │   │ Follows     │ │ Follows     │    │                │
│         │   │ Director's  │ │ Director's  │    │                │
│         │   │ Instructions│ │ Instructions│    │                │
│         │   └──────┬──────┘ └──────┬──────┘    │                │
│         └──────────┼───────────────┼───────────┘                │
│                    │               │                             │
│                    └───────┬───────┘                             │
│                            │                                     │
│                     Generated Dialogue                           │
│                            │                                     │
│                ┌───────────▼────────────┐                       │
│                │   HUB DISTRIBUTION      │                       │
│                │  (mac_hub_bridge.py)    │                       │
│                │   Port 8766             │                       │
│                └───────────┬────────────┘                       │
│                            │                                     │
│                     WebSocket to Devices                        │
│                            │                                     │
│              ┌─────────────┼─────────────┐                      │
│              ▼             ▼             ▼                      │
│         [Gameboy 1]   [Gameboy 2]   [Other Devices]             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Scene Server (scene_server.py)
**Previously:** enhanced_scene_manager_client.py

```python
# Main entry point - FastAPI server on port 8787
# Endpoints:
POST /scene/start     - Initialize new scene
POST /scene/stop      - Stop current scene
POST /scene/continue  - Continue stopped scene
GET  /scene/status    - Get current status

# Creates and manages:
- SceneOrchestrator instance
- Character agents (Laura, Claude)
- WebSocket connections to hub
```

### 2. Scene Orchestrator (orchestrator.py)
**Previously:** enhanced_pipeline_orchestrator.py

```python
class SceneOrchestrator:
    """Coordinates the entire scene flow"""

    async def run_scene(topic, participants, max_turns):
        # Step 1: Director initializes scene
        scene_plan = await director.initialize_scene(topic, participants)

        # Step 2: Main loop for each turn
        for turn in range(1, max_turns+1):
            # Director decides EVERYTHING first
            decision = await director.direct_next_turn(history)

            # Get the chosen agent
            agent = agents[decision.chosen_speaker]

            # Pass Director's instructions to agent
            response = await agent.generate_response(
                prompt=current_prompt,
                director_instruction=decision.format_for_agent()
            )

            # Distribute to devices
            await hub.distribute(response)
```

### 3. Scene Director Agent (scene_director_agent.py)
**NEW - The actual brain**

```python
class SceneDirectorAgent:
    """Makes all creative decisions"""

    async def initialize_scene(topic, participants):
        # Analyzes topic and creates dramatic arc
        # Returns: first_speaker, character_objectives, escalation_plan

    async def direct_next_turn(conversation_history):
        # THE CORE METHOD - called BEFORE each agent speaks
        # Decides:
        #   - chosen_speaker: who talks next
        #   - instruction: "Defend yourself but reveal guilt"
        #   - emotional_tone: "flustered"
        #   - scene_note: "Time for plot twist"
        # Returns: DirectorDecision object
```

### 4. Character Agents (claude_agent.py, laura_agent.py)
**Previously:** enhanced_claude_agent.py, enhanced_laura_agent.py

```python
class CharacterAgent:
    """Actors that follow Director's instructions"""

    # System blocks (CACHED - static personality):
    - Character definition
    - JSON output format
    - Basic rules

    # Messages array (dynamic):
    - Director's instruction for this turn
    - Conversation history
    - Current prompt

    async def generate_response(prompt, director_instruction):
        # Combines static character with dynamic direction
        # Returns: dialogue text + next_speaker (if not overridden)
```

### 5. API Adapter (anthropic_adapter.py)
**Previously:** enhanced_anthropic_adapter.py

```python
class AnthropicAdapter:
    """Handles all LLM API calls with caching"""

    async def generate_response(messages, system_blocks):
        # system_blocks: cached static content
        # messages: dynamic per-request content
        # Returns: API response with cache metrics
```

## Data Flow for Each Turn

```
1. ORCHESTRATOR: "Time for turn 3"
        │
        ▼
2. DIRECTOR: Analyzes scene state
   - Reviews conversation history
   - Checks dramatic arc progress
   - Makes decision: {speaker: "laura", instruction: "accuse claude of lying"}
        │
        ▼
3. ORCHESTRATOR: Gets Laura agent
        │
        ▼
4. LAURA AGENT: Receives:
   - Director's instruction: "accuse claude of lying"
   - Conversation history
   - Prompt: "Generate next dialogue"
        │
        ▼
5. ANTHROPIC API:
   - Cached: Laura's personality (4200 tokens)
   - Dynamic: Director instruction + history
        │
        ▼
6. LAURA: "You're lying! I saw you take my water bottle!"
        │
        ▼
7. HUB DISTRIBUTION: Send to gameboys
```

## System Instructions Architecture

### Director Agent:
```
Planning Phase:
"You are a TV show director planning a comedic scene.
Output only valid JSON for scene structure."

Per-Turn Phase:
"You are a TV director making specific scene decisions.
Output JSON with speaker choice, emotional tone, and instructions."
```

### Character Agents:
```
STATIC (Cached):
"You are [CHARACTER] in a comedic scene.
[Full personality definition]
[JSON output format]
[Basic rules]"

DYNAMIC (Per-turn in messages):
"[DIRECTOR'S INSTRUCTION]
Emotional tone: defensive
Specific direction: Reveal you knew all along
[END INSTRUCTION]

Recent conversation:
CLAUDE: Where's my cushion?
LAURA: I don't know what you mean..."
```

## Cache Strategy

```
┌─────────────────────────────────────┐
│         CACHED CONTENT              │
│  (Stays same across all turns)      │
├─────────────────────────────────────┤
│ • Character personalities           │
│ • JSON format instructions          │
│ • Sprite galleries (Laura)          │
│ • Basic performance rules           │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      DYNAMIC CONTENT                │
│   (Changes every turn)              │
├─────────────────────────────────────┤
│ • Director's instruction            │
│ • Conversation history              │
│ • Current prompt                    │
│ • Turn-specific context             │
└─────────────────────────────────────┘
```

## Key Architecture Principles

1. **Director-First**: Director decides BEFORE agents speak, not after
2. **Clear Separation**: Director = brain, Agents = actors, Orchestrator = coordinator
3. **Visible Value**: Director logs reasoning for every decision
4. **Proper Caching**: Static personality cached, dynamic instructions not
5. **No Fallbacks**: Errors fail properly, no "theatrical" responses

## File Structure (Cleaned)

```
scene_manager/
├── scene_server.py          # Main FastAPI entry (was enhanced_scene_manager_client)
├── orchestrator.py          # Turn coordination (was enhanced_pipeline_orchestrator)
├── scene_director_agent.py  # NEW - The brain
├── anthropic_adapter.py     # API calls (was enhanced_anthropic_adapter)
├── agents/
│   ├── claude_agent.py      # Claude character (was enhanced_claude_agent)
│   ├── laura_agent.py       # Laura character (was enhanced_laura_agent)
│   └── agent_base.py        # Base class (was scene_agent_base)
├── utils/
│   ├── context_loader.py    # Character context
│   └── image_optimizer.py   # Sprite handling
└── archive/                  # All old/test files
```

## Example Turn Execution

```python
# Turn 3: Director sees Laura just accused Claude

decision = DirectorDecision(
    chosen_speaker="claude",
    instruction="Deny it with aristocratic indignation, but accidentally reveal you know exactly which cushion she means",
    emotional_tone="offended but guilty",
    scene_note="This is the reveal moment"
)

# Claude receives:
"[DIRECTOR'S INSTRUCTION]
Emotional tone: offended but guilty
Specific direction: Deny it with aristocratic indignation, but accidentally reveal you know exactly which cushion she means
Scene note: This is the reveal moment
[END INSTRUCTION]"

# Claude generates:
"How DARE you accuse me of... wait, you mean the blue microfiber one with the memory foam? I've never seen it."
```

## Success Metrics

1. **Director Control**: Every turn shows Director's decision and reasoning
2. **Cache Efficiency**: <500ms response time with 4200 token cache hits
3. **Scene Quality**: Coherent dramatic arcs with proper pacing
4. **No Agent Autonomy**: Agents never pick speakers without Director approval
5. **Clear Value**: Can demonstrate why Director improves over random alternation