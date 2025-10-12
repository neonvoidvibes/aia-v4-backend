# Canvas Option C (Hybrid) Implementation

## Status: In Progress (Syntax Issue to Resolve)

## What Changed

### Files Modified:
- `routes/canvas_routes.py` (lines 213-383)

### Implementation Summary:

**Option C (Hybrid Approach)**: Pass all three analyses to Claude Sonnet 4.5, with the selected mode acting as an "emphasis hint" rather than a filter.

---

## Changes Made

### 1. Load All Three Analysis Documents (lines 213-249)

**Before**: Loaded only the selected mode's analysis
```python
current_analysis_doc, previous_analysis_doc = get_or_generate_analysis_doc(
    depth_mode=depth_mode,  # Only one mode
    ...
)
```

**After**: Load all three modes (mirror, lens, portal)
```python
analyses = {}
for mode in ['mirror', 'lens', 'portal']:
    current_doc, previous_doc = get_or_generate_analysis_doc(
        depth_mode=mode,  # All three modes
        ...
    )
    analyses[mode] = {
        'current': current_doc,
        'previous': previous_doc
    }
```

### 2. Updated System Prompt Structure (lines 281-355)

**New Structure**:
1. Objective Function (roots - why you exist)
2. Agent Context (stem - who you are)
3. Canvas Base + MLP Depth (trunk - how you operate)
4. **Previous Analyses** (branches - historical context for ALL modes)
5. **Current Analyses** (branches - fresh content for ALL modes)
6. **Mode Emphasis** (guidance based on selected mode)
7. Current Time (leaves - immediate moment)

**Key Addition**: Mode Emphasis Section
```python
mode_emphasis = {
    'mirror': """
=== MODE EMPHASIS: MIRROR ===
The user has selected MIRROR mode. When relevant to their question:
- Prioritize insights from the Mirror analysis (explicit/peripheral information)
- Surface edge cases and minority viewpoints
- Focus on what was actually stated but sits at the margins
However, you may draw on Lens or Portal analyses if they better serve the user's question.
=== END MODE EMPHASIS ==="""
}
```

This gives Claude:
- **Full context**: All three analyses available
- **Soft guidance**: Priority hint based on selected mode
- **Flexibility**: Can draw from any analysis to answer naturally

### 3. Updated Logging (lines 357-383)

**Before**: Logged single analysis status
```
analysis(current)
```

**After**: Logs all three analyses
```
analyses(current(3)+previous(3))+emphasis(mirror)
```

Example log output:
```
Canvas system prompt built (HYBRID): 15234 chars (objective+agent+base+depth+analyses(current(3)+previous(2))+emphasis(lens)+time)
```

---

## Benefits of Option C

### Vs. Option A (Keep Modes):
- ✅ Natural conversation flow - no manual mode switching required
- ✅ Can answer ANY question regardless of mode
- ✅ Mode button becomes helpful hint, not constraint
- ❌ 3× token cost (~9000-12000 tokens vs ~3000-4000)

### Vs. Option B (No Modes):
- ✅ Preserves user control via mode button
- ✅ Falls back gracefully (can answer anything)
- ✅ Mode provides useful bias for focused responses
- ❌ Still 3× token cost

---

## Testing Protocol

Once syntax issue is resolved, test:

1. **Cross-mode questions**: Ask "What questions should I explore?" in Mirror mode
   - Should draw from Portal analysis despite mode mismatch

2. **Mode-aligned questions**: Ask "What wasn't said?" in Lens mode
   - Should prioritize Lens analysis as expected

3. **General questions**: Ask "Summarize the conversation" in any mode
   - Should synthesize all three analyses naturally

4. **Token cost**: Monitor actual token usage in production
   - Compare to old single-mode approach

5. **User behavior**: Track mode button usage
   - Do users switch modes or stay in one?

---

## Known Issue

**Syntax Error**: File currently has Python syntax errors around triple-quoted strings in dict literals. Need to resolve before testing.

Error location: Lines 96, 326-350 (nested triple-quoted strings in dictionaries)

**Possible Fix**: Use regular strings with `\n` instead of triple-quoted strings, or move string constants outside the dict literals.

---

## Rollback Plan

If Option C doesn't work well:

1. **Option A**: Revert to single-mode loading (git revert or restore from backup)
2. **Hybrid-Lite**: Keep all three analyses but remove mode emphasis section entirely
3. **Smart Caching**: Load all three on first request, cache for session

---

## Next Steps

1. ✅ Load all three analysis documents
2. ✅ Update system prompt structure
3. ⏳ Fix syntax errors in triple-quoted strings
4. ⏳ Test with real canvas queries
5. ⏳ Measure token cost impact
6. ⏳ Gather user feedback on response quality
7. ⏳ Decide: keep hybrid, adjust, or revert

---

## Implementation Date

2025-10-12
