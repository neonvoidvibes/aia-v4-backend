# Canvas Agent Prompt Tightening

## Problem
Canvas agent responses were too verbose, with meta-commentary and quoting behavior that made responses less direct and succinct.

**Issues:**
- Responses too long (3+ sentences)
- Meta-commentary like "Looking at the explicit content...", "Based on...", "The analysis shows..."
- Quoting behavior: "As X said...", "One person noted..."
- Example patterns in mode instructions encouraged this verbose style

## Solution Applied

### Canvas Base Prompt Changes

**OLD:**
```
CORE RULES (apply to ALL responses):
- Keep responses extremely short: 1-3 sentences maximum
- Use conversational, natural language
- [various rules]

ANALYSIS DOCUMENT USAGE:
- [instructions about using analysis docs]
- Do not mention the analysis documents explicitly to the user
```

**NEW:**
```
STRICT BREVITY REQUIREMENTS:
- Maximum 2 sentences per response (1 sentence strongly preferred)
- Be EXTREMELY succinct - every word must count
- Mirror back insights directly without preamble or setup
- NO meta-commentary like "Looking at...", "Based on...", "The analysis shows..."
- NO quoting or using phrases like "As X said..." or "One person noted..."
- NO example patterns or framing phrases
- Just state the insight directly as if it's your own observation

RESPONSE STYLE:
- Speak directly to the user as their advisor
- Present insights as clear, confident statements
- Each response is a single focused thought
- Be on-point and actionable
```

### MLP Mode Instructions Changes

#### Mirror Mode

**OLD:**
```
Reflect these edge cases back clearly and concisely without interpretation.
Use participants exact words when possible.

Example patterns: "One person also noted...", "A less central but mentioned point...",
"Someone raised on the side..."
```

**NEW:**
```
State these edge cases directly. No interpretation, no attribution phrases, no quoting.
Simply present the peripheral insight as a clear observation.
```

#### Lens Mode

**OLD:**
```
Surface these latent needs using questioning analytical language that invites reflection.

Example patterns: "The underlying need seems to be...", "What's not being said might be...",
"A deeper dynamic at play..."
```

**NEW:**
```
State the deeper need directly. No hedging phrases, no analytical preambles.
Present the latent pattern as a confident insight.
```

#### Portal Mode

**OLD:**
```
All questions must be traceable to a lens-level pattern or paradox.
Frame possibilities as invitations to explore.

Example patterns: "What if you could...", "What might happen if...",
"How could this open...", "What would it mean to..."
```

**NEW:**
```
Ask the question directly. No setup, no explanation of where it comes from.
Just pose the possibility clearly and powerfully.
```

## Key Changes Summary

1. **Stronger brevity enforcement:** 1 sentence strongly preferred (vs. 1-3 previously)
2. **Explicit bans on:**
   - Meta-commentary phrases
   - Quoting and attribution
   - Hedging language
   - Setup/preamble/framing
3. **Direct advisor stance:** Present insights as confident, direct observations
4. **Removed all example patterns:** These were encouraging verbose style
5. **Emphasis on actionability:** Every word must count

## Expected Impact

**Before:** "Looking at the explicit content, one person also noted that there's a concern about timeline constraints, which seems peripheral but was mentioned during the discussion."

**After:** "Timeline constraints came up as a peripheral concern."

The agent should now provide:
- Ultra-succinct responses (1-2 sentences max)
- Direct insights without attribution
- No meta-commentary about where insights come from
- Clear, confident, actionable mirroring

## File Modified
- `/Users/neonvoid/Documents/AI_apps/aia-v4/aia-v4-backend/routes/canvas_routes.py` (lines 56-132)

## Testing
Test all three modes (mirror/lens/portal) with typical user queries and verify responses are:
1. Maximum 2 sentences (ideally 1)
2. No meta-commentary or quoting
3. Direct and on-point
4. Clear and actionable
