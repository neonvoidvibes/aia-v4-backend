# Source Differentiation Fix - Preserving Multiple Breakout/Group Signals

## Problem Statement

When users select "Settings > Memory > Transcripts > Groups: breakout" with multiple breakout groups (e.g., breakout_1 and breakout_2), the system was **losing differentiation signal** at each analysis level:

**a) MLP agents** - SAW differentiation (labels existed) but had NO instructions to PRESERVE it ❌
**b) MLP outputs** - BLENDED sources together without attribution ❌
**c) Canvas agent** - Had NO WAY to recover differentiation from blended analysis ❌

**User concern:** "we don't want the canvas agent to reflect back as if they were handed *one* information stream because we'd lose too much differentiation signal"

## Root Cause

### What Data Structure Looked Like

**MLP agents received:**
```
=== BREAKOUT EVENTS TRANSCRIPTS ===
--- Breakout Event: breakout_1 ---
{transcripts from breakout 1}

--- EVENT SEPARATOR ---

--- Breakout Event: breakout_2 ---
{transcripts from breakout 2}
=== END BREAKOUT EVENTS TRANSCRIPTS ===
```

**The labels existed** - but there were **NO instructions** telling agents to:
- Maintain source attribution in their analysis
- Compare/contrast patterns between sources
- Preserve differentiation signal through the chain

**Result:** MLP agents would write analysis like:
> "The group discussed resource constraints. Participants expressed concerns about timelines..."

**Lost information:** Which breakout said what? Were they aligned or divergent? Critical signal gone!

## Solution Applied

### 1. MLP Agent Instructions (canvas_analysis_agents.py lines 72-89)

Added **SOURCE DIFFERENTIATION REQUIREMENT** section after transcript data:

```python
transcript_section = f"""
=== TRANSCRIPT DATA ===
{transcript_content}
=== END TRANSCRIPT DATA ===

CRITICAL: The transcript data above may contain MULTIPLE SOURCES with distinct labels
(e.g., different breakout groups, events, or transcript files). Each source is clearly
marked with headers like "--- Breakout Event: X ---" or "--- START Transcript Source: Y ---".

SOURCE DIFFERENTIATION REQUIREMENT:
When analyzing transcripts from multiple sources, you MUST maintain clear attribution:
- Identify patterns or themes WITHIN each source
- Compare/contrast patterns BETWEEN sources when relevant
- Always specify which source(s) your observations come from
- Use source labels naturally in your narrative (e.g., "In breakout_1, participants
  focused on X, while breakout_2 emphasized Y")
- If all sources show the same pattern, note this explicitly: "Across all breakouts..."
- Preserve the differentiation signal - don't blend sources into a homogeneous "the group"

If only ONE source is present, you may refer to "the conversation" or "the group" naturally.
But with MULTIPLE sources, maintain their distinctness throughout your analysis.
"""
```

**Impact on MLP Agents:**

All three agents (Mirror, Lens, Portal) now:
- Recognize when multiple sources are present
- Maintain source attribution throughout their analysis
- Compare/contrast patterns between sources
- Preserve differentiation signal in their outputs

### 2. Canvas Agent Instructions (canvas_routes.py lines 84-90)

Added **SOURCE DIFFERENTIATION** section to canvas base prompt:

```python
SOURCE DIFFERENTIATION:
- Analysis documents may contain insights from MULTIPLE SOURCES (e.g., different breakout
  groups, transcript files, or events)
- When responding, preserve source differentiation if it matters to the user's question
- If asked about "breakout 1" specifically, draw only from that source's insights
- If asked about differences between groups, contrast them naturally
- If the pattern is universal across sources, state it confidently without qualifying
- NEVER blend sources into "the group" if differentiation is relevant to the insight
```

**Impact on Canvas Agent:**

Canvas can now:
- Understand that analysis documents contain differentiated sources
- Respond to source-specific questions ("what did breakout 1 focus on?")
- Contrast sources when asked ("how did the breakouts differ?")
- Preserve differentiation when mirroring insights back to user

## Expected Behavior Changes

### Before Fix (Lost Differentiation):

**MLP Mirror Analysis:**
> "The group discussed resource constraints extensively. Participants expressed concerns about timeline feasibility and budget limitations. Several people mentioned staffing challenges."

**Canvas Response to "What did each breakout focus on?":**
> "Resource constraints and timeline concerns came up across the conversation."

❌ **No way to know** which breakout said what!

### After Fix (Preserved Differentiation):

**MLP Mirror Analysis:**
> "In breakout_1, the conversation centered on resource constraints, with multiple participants returning to budget limitations and staffing gaps. By contrast, breakout_2 focused primarily on timeline feasibility, with participants emphasizing delivery deadlines and milestone dependencies. Both groups mentioned project scope, but breakout_1 framed it as a resource allocation problem while breakout_2 viewed it through a scheduling lens."

**Canvas Response to "What did each breakout focus on?":**
> "Breakout 1 centered on resource constraints and staffing, while breakout 2 emphasized timeline feasibility and delivery milestones."

✅ **Differentiation preserved** through the entire chain!

## Testing Scenarios

### Scenario 1: Two Breakouts with Different Themes

**Setup:**
- Breakout_1 discusses technical architecture
- Breakout_2 discusses user experience
- User selects "Groups: breakout"

**Expected MLP Output:**
- Mirror: Should clearly attribute which breakout discussed which theme
- Lens: Should identify different latent needs per breakout
- Portal: Should ask questions tailored to each breakout's focus

**Expected Canvas Behavior:**
- If asked "What did breakout 1 focus on?" → Technical architecture only
- If asked "How did the breakouts differ?" → Contrast tech vs UX focus
- If asked general question → Synthesize but maintain source awareness

### Scenario 2: Two Breakouts with Convergent Themes

**Setup:**
- Both breakouts discuss resource constraints but from different angles
- User selects "Groups: breakout"

**Expected MLP Output:**
- "Across both breakouts, resource constraints dominated the conversation. However, breakout_1 approached this from a budget perspective while breakout_2 focused on staffing capacity."

**Expected Canvas Behavior:**
- Recognizes convergence but preserves nuance about different angles

### Scenario 3: Single Source (Control)

**Setup:**
- Only one transcript file selected
- User selects "Listen: latest"

**Expected Behavior:**
- MLP agents naturally refer to "the conversation" or "the group"
- No awkward source attribution needed
- System works as before for single-source scenarios

## Files Modified

### 1. `/Users/neonvoid/Documents/AI_apps/aia-v4/aia-v4-backend/utils/canvas_analysis_agents.py`
- **Lines 72-89:** Added SOURCE DIFFERENTIATION REQUIREMENT to transcript section
- **Impact:** All three MLP agents (Mirror, Lens, Portal) receive these instructions

### 2. `/Users/neonvoid/Documents/AI_apps/aia-v4/aia-v4-backend/routes/canvas_routes.py`
- **Lines 84-90:** Added SOURCE DIFFERENTIATION section to canvas base prompt
- **Impact:** Canvas agent understands analysis documents contain differentiated sources

## Verification Steps

1. **Enable breakout mode:**
   - Settings > Memory > Transcripts > Groups: breakout

2. **Verify MLP agents see labels:**
   - Check backend logs for: `=== BREAKOUT EVENTS TRANSCRIPTS ===`
   - Confirm multiple `--- Breakout Event: X ---` labels appear

3. **Verify MLP analysis preserves differentiation:**
   - Click sparkles icon to regenerate analysis
   - Check S3 analysis documents for source attribution
   - Look for phrases like "In breakout_1..." and "In breakout_2..."

4. **Verify canvas responses maintain differentiation:**
   - Ask: "What did breakout 1 focus on?"
   - Ask: "How did the breakouts differ?"
   - Ask: "What patterns emerged across all groups?"
   - Confirm responses distinguish sources appropriately

## Architecture Alignment

This fix aligns with the principle that **MLP agents are comprehensive analyzers** while **canvas agents are succinct responders**:

- **MLP agents:** Do the heavy lifting of analyzing and maintaining source differentiation in their documents (1500-2000 tokens)
- **Canvas agent:** Draws on differentiated analysis to provide brief, targeted responses (1-2 sentences) that respect source boundaries

The differentiation signal flows:
```
Raw Transcripts (labeled)
  → MLP Agents (maintain labels in analysis)
    → MLP Documents (contain attributed insights)
      → Canvas Agent (responds with source awareness)
        → User (receives differentiated insights)
```

## Rollout

Changes take effect immediately for **new analysis generations**. Existing cached analyses will update when:
- User clicks sparkles icon (force refresh)
- User enters a new meeting context (clearPrevious=true)
- Cache expires (15 minute TTL)

No migration needed - this is a prompt-only change.
