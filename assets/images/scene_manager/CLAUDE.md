# Scene Manager Visual Feedback System - LOCAL GAMEBOY DEMO

## ⚠️ CRITICAL: SYSTEM BOUNDARIES ⚠️

**THIS DIRECTORY: `/Users/lauras/Desktop/laura/scene_manager/`**
- **Purpose**: Local testing with physical Raspberry Pi gameboy devices
- **Ports**: 9000 (scene_server), 9001 (gameboy_hub), 9002 (claude_agent), 9003 (laura_agent)
- **NOT connected to ngrok**
- **This is the ACTIVE WORK DIRECTORY for gameboy testing**

**DO NOT CONFUSE WITH:**
- `/Users/lauras/Desktop/laura/scene_manager_repo/hf_clean/` - Web demo with ngrok (ports 7860, 8001, 8002)
- `/Users/lauras/Desktop/laura/scene_manager_repo/scene_manager/` - Repository version (DO NOT EDIT)

**ALWAYS WORK IN**: `/Users/lauras/Desktop/laura/scene_manager/` for gameboy demo development

## Project Overview
Enhanced scene manager for multi-agent improv conversations with visual feedback on Raspberry Pi gameboy devices.

## Notable Bugs & Comedy Gold

### The Carson Appeal Incident (September 26, 2025)
During a heated scene about Claude stealing Laura's micro napping pad, Laura got so frustrated she broke the fourth wall:

**Laura:** "Superior reasoning? You're literally a gray box sitting on a stolen cushion arguing about property law - Carson, tell him!"

Laura attempted to call Carson (the user) as `next_speaker`, causing the scene manager to crash looking for a non-existent agent. While technically a bug, it was comedically perfect - Laura appealing to the human authority figure to settle the AI dispute. Fixed by adding stricter validation to ensure `next_speaker` must be in the participants list.

## Current Architecture
- **6-turn scenes** with 2-3 participants
- **Scene Director**: Provides shared context + private directions via LLM
- **Character Agents**: Claude (aristocratic AI) and Laura (chaotic energy)
- **Display Devices**: Raspberry Pi in GPi Case2 (640x480 screen)

## Visual System Requirements

### Laura (Gameboy 2)
- **Anime-style sprites** (blue polo Best Buy employee girl)
- **Two states**: talking/ and idle/ folders with PNG sprites
- **Dynamic selection**: Agent picks speaking + idle image each turn
- **Cached sprite gallery**: ~1,570 tokens cached, 327 per turn

### Claude (Gameboy 1)
- **SVG asterisk logo** (Anthropic's wonky star)
- **Three moods**: neutral, helpful, snarky
- **Audio-reactive animation**: Scale tied to speech amplitude
- **Low compute requirement**: Must run smoothly on Pi Zero

## Claude SVG Animation Goals

### Core Requirements
1. **Minimal CPU usage** - Pi Zero compatible (~30fps max)
2. **Audio synchronization** - Visual feedback matches TTS playback
3. **Character personality** - Animations reflect aristocratic condescension
4. **Smooth transitions** - No jarring movements

### Animation States

#### Neutral (Default)
- Gentle "breathing" scale (0.98-1.02)
- Slow rotation (~1 deg/sec)
- Orange color (#FF6600)

#### Helpful (Engaged)
- Warm glow effect (opacity pulse)
- Slightly larger scale (1.05)
- Lighter orange (#FF9933)

#### Snarky (Condescending)
- Sharp, tilted position (5-10 degrees)
- Quick scale snaps on emphasis
- Darker orange (#CC4400)

#### Speaking (Audio-reactive)
- Scale mapped to amplitude (0.9-1.1 range)
- Bounces with speech rhythm
- Returns to base state between phrases

## Evaluation Criteria for Animation Script

### Performance
- [ ] Runs at 25-30fps on Pi Zero W
- [ ] CPU usage stays under 50%
- [ ] No memory leaks over 6-turn scene
- [ ] Smooth transitions without stuttering

### Audio Integration
- [ ] Correctly samples audio amplitude
- [ ] Responsive to speech patterns
- [ ] Proper silence detection
- [ ] Synchronizes with TTS timing

### Visual Quality
- [ ] Animations match Claude's personality
- [ ] Smooth interpolation between states
- [ ] No jarring transitions
- [ ] Maintains readability at 640x480

### Code Quality
- [ ] Clean state management
- [ ] Proper error handling
- [ ] Configurable parameters
- [ ] Easy mood switching

## Implementation Notes

### SVG Structure
```xml
- Single path element (complex asterisk shape)
- ViewBox: 1104.33 × 1080
- No stroke, fill only
- Transform origin: center
```

### Recommended Approach
1. CSS transforms only (no path manipulation)
2. RequestAnimationFrame for smooth updates
3. Web Audio API or amplitude sampling
4. State machine for mood transitions

### Resource Constraints
- **RAM**: ~512MB available
- **GPU**: Limited/none
- **Display**: 640x480 LCD
- **Python**: Preferred for consistency

## Testing Checklist
1. Load SVG and verify rendering
2. Test each mood state independently
3. Verify audio amplitude detection
4. Check CPU usage during speech
5. Test transitions between moods
6. Run full 6-turn scene
7. Monitor for memory leaks

## Success Metrics
- Adds personality without distraction
- Enhances scene immersion
- Maintains performance on Pi hardware
- Provides clear visual feedback
- Matches character personality