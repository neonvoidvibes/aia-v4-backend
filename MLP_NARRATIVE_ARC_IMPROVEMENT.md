# MLP Analysis Agents - Narrative Arc Improvements

## User Questions Answered

### a) Will MLP agents understand "groups: breakout" for themselves?
**YES** - Lines 408-427 in `canvas_analysis_agents.py` show full breakout support:
- MLP agents receive transcripts with headers: `=== BREAKOUT EVENTS TRANSCRIPTS ===`
- Each breakout labeled: `--- Breakout Event: {event_id} ---`
- Database query filters for `type='breakout'` events (excluding `visibility_hidden=true`)
- All breakout transcripts are read and passed to analysis agents

### b) Will their outputs reflect this?
**YES** - The MLP agents see structured headers showing which transcripts are from breakouts, so their analysis naturally incorporates this context. The transcript data includes event IDs and separators.

### c) Will the canvas agent understand this?
**YES** - Canvas agent receives the MLP analysis documents which contain insights derived from the breakout transcripts. Note: Canvas doesn't see raw transcripts - only the pre-analyzed MLP insights.

## Problems Identified

### 1. Consistent Headers
**Status:** Headers were already consistent across all three modes:
- `# Mirror/Lens/Portal Analysis: [description]`
- `## Surface Level: [description]`
- `## Deep Level: [description]`
- `## Most Pertinent Observation/Question`

No changes needed.

### 2. Mirror Agent Too Concrete/Repetitive
**Problem:** Lines 101, 112 had formulaic patterns causing mechanical repetition:
- "You're discussing...", "The group agrees that...", "Several of you have mentioned..."
- "One participant also noted...", "A less discussed but mentioned point..."
- This created monotonous "you say", "you continue", "you expand" style

### 3. Lack of Narrative Arc
**Problem:** All three agents had:
- Bullet-pointed instructions creating disconnected observations
- "Language patterns:" sections encouraging formulaic repetition
- No guidance on flowing narrative structure
- Mechanical tone rather than engaging storytelling

## Solutions Applied

### Global Improvements (All Three Agents)

**1. Added "NARRATIVE GUIDANCE" sections:**
- Emphasize flowing prose over formulaic patterns
- Encourage varied sentence structure
- Request natural transitions between themes
- Build coherent narrative arcs

**2. Removed "Language patterns:" sections:**
- Eliminated formulaic templates like "The group agrees that..."
- Replaced with anti-patterns: "Don't use... Instead..."
- Encouraged natural language variation

**3. Added narrative metaphors:**
- Mirror: "craft a readable story"
- Lens: "detective story that uncovers hidden patterns"
- Portal: "journey of inquiry"

### Mirror Agent Specific Changes

**Tone guidance:**
```
OLD: Neutral, observational, present tense, no interpretation.
NEW: Neutral, observational, present tense. Craft a flowing narrative that moves
naturally through themes rather than mechanical repetition. Vary your language and
sentence structure to create a readable story of what was said.
```

**Surface Level:**
```
OLD: [For EACH central theme (aim for 3-5):
- State the theme clearly using participants' exact formulations
- Quote multiple speakers expressing this theme
Language patterns: "You're discussing...", "The group agrees that..."

NEW: Weave together 3-5 central themes into a flowing narrative. For each theme:
- State it clearly using participants' exact formulations
- Quote multiple speakers naturally within the narrative

Write as a cohesive narrative, not as separate bullet points. Vary your phrasing -
avoid starting every sentence with "The group..." or "You're discussing..." Instead,
let the themes flow naturally: describe what emerges, what recurs, what connects.
```

**Deep Level:**
```
OLD: Language patterns: "One participant also noted...", "A less discussed but mentioned point..."

NEW: Create a narrative that shows how these edge cases relate to the whole. Don't use
formulaic openings like "One participant also noted..." - instead, integrate them
naturally: describe when they appeared, how they contrasted, what they revealed.
```

**New guidelines:**
- Write in flowing narrative prose, not mechanical patterns
- Vary sentence structure and openings significantly
- Create transitions between themes for readability
- This is pure reflection crafted as a coherent story

### Lens Agent Specific Changes

**Tone guidance:**
```
OLD: Analytical, questioning, surface paradoxes and tensions.
NEW: Analytical, questioning, revealing. Craft a detective story that uncovers hidden
patterns and unspoken needs. Build tension and insight through a narrative arc that
connects disparate clues into coherent understanding.
```

**Surface Level:**
```
OLD: Language patterns: "There's a pattern emerging around...", "Several comments suggest..."

NEW: Write as a coherent investigation, not separate observations. Vary your analytical
voice - avoid formulaic openings like "There's a pattern emerging..." or "Several
comments suggest..." Instead, let patterns reveal themselves naturally: describe what
connects, what recurs differently, what tensions appear.
```

**Deep Level:**
```
OLD: Language patterns: "The underlying need seems to be...", "What's not being said directly is..."

NEW: Create a narrative that shows how surface patterns reveal deeper needs. Don't
mechanically repeat phrases like "The underlying need seems to be..." - instead, build
your case naturally: show how evidence accumulates, how contradictions point to hidden
dynamics, how avoidances reveal priorities.
```

**New guidelines:**
- Write as an unfolding investigation, not a checklist
- Build connections between patterns organically
- Create narrative momentum - each insight leads naturally to the next
- Let paradoxes and tensions drive the narrative forward
- This is rigorous interpretation crafted as a coherent investigation

### Portal Agent Specific Changes

**Tone guidance:**
```
OLD: Visionary, possibility-oriented, future tense welcomed. Frame as invitations.
NEW: Visionary, possibility-oriented, invitational. Craft a journey of inquiry that
builds from broad possibilities to specific interventions. Create narrative flow through
your questions - each opens naturally from the previous, building momentum toward
transformation.
```

**Surface Level:**
```
OLD: Language patterns: "What if you could...", "How might...", "What would it mean to..."

NEW: Write questions that build on each other naturally, creating an arc of exploration.
Vary your question openings significantly - don't mechanically repeat "What if..." or
"How might..." Instead, mix structures: direct questions, compound questions, conditional
questions, exploratory questions. Let each question set feel like a distinct gateway
into new territory.
```

**Deep Level:**
```
OLD: Language patterns: "What if you...", "How might this cascade into...", "What leverage points exist for..."

NEW: Write questions that create a sense of modeling - as if you're thinking through
interventions in real-time. Vary your question styles: some short and direct, others
long and exploratory. Build question chains that feel like you're following a thread
of possibility to its logical extensions.
```

**New guidelines:**
- Write questions that flow from one to another organically
- Build complexity gradually - simple to nuanced
- Create thematic threads that connect question sets
- Vary question structure dramatically - avoid repetitive openings
- Create narrative momentum through your questions
- This is strategic inquiry crafted as a visionary journey

## Expected Impact

### Before (Mechanical Style)
**Mirror:**
> "You're discussing timeline constraints. The group agrees that this is a concern. Several of you have mentioned resource allocation. You continue to express worry about deadlines. One participant also noted budget issues."

**Lens:**
> "There's a pattern emerging around resource scarcity. Several comments suggest an underlying tension about priorities. The energy shifts when discussing leadership decisions."

**Portal:**
> "What if you could improve resource allocation? How might that shift the paradigm? What would it mean to restructure timelines? What becomes possible when you address these concerns?"

### After (Narrative Arc Style)
**Mirror:**
> "Timeline constraints dominate the conversation, with multiple voices returning to this theme through different lenses. Resource allocation emerges as both symptom and cause - the group circles this repeatedly, each iteration adding nuance. What begins as surface-level deadline concerns deepens into questions about capacity and prioritization. Budget issues surface late but connect back, revealing how financial constraints shape everything discussed earlier."

**Lens:**
> "Resource scarcity isn't just mentioned - it permeates the entire exchange, shaping what can and cannot be said. As the conversation progresses, a pattern emerges: whenever leadership decisions approach discussion, the energy shifts defensively. This suggests something protected, perhaps an unspoken agreement to avoid naming authority dynamics directly. The repeated cycling back to timelines and budgets may serve as safer proxies for harder questions about decision-making power and strategic direction."

**Portal:**
> "If resource scarcity is the presenting symptom, what becomes possible by examining the underlying allocation philosophy? Could restructuring decision authority create space for more adaptive resource flows? What leverage points exist at the intersection of timeline flexibility and capacity building? As these questions connect, a larger possibility emerges: what would it look like to move from reactive resource management to proactive capacity design, and what would need to shift first to make that leap viable?"

## Testing

To test these improvements:

1. **Generate fresh analyses** for an agent with multiple transcripts or breakout data
2. **Check for:**
   - Varied sentence openings (no repetitive "You're discussing..." or "There's a pattern...")
   - Natural transitions between themes
   - Coherent narrative flow
   - Questions that build on each other (Portal)
   - Evidence woven throughout, not listed (Lens)
   - Quotes integrated smoothly, not mechanically attributed (Mirror)
3. **Verify breakout mode** works by selecting "groups: breakout" and checking backend logs for:
   - `=== BREAKOUT EVENTS TRANSCRIPTS ===` headers
   - Proper event ID labels
   - Analysis reflecting multiple breakout contexts

## Files Modified

- `/Users/neonvoid/Documents/AI_apps/aia-v4/aia-v4-backend/utils/canvas_analysis_agents.py`
  - Lines 80-136: Mirror agent prompt
  - Lines 139-201: Lens agent prompt
  - Lines 204-267: Portal agent prompt

## Rollout

Changes take effect immediately for new analysis generations. Existing cached analyses will be replaced when:
- User clicks sparkles icon (force refresh)
- User enters a new meeting context (clearPrevious=true)
- Cache expires (15 minute TTL)
