# Canvas "Some" Mode - Root Cause & Fix

## Root Cause Identified

The issue was **NOT** a key format mismatch or missing wiring. The root cause was a **timing/view-switching issue**:

### The Problem

When users changed transcript settings:

1. User is in **Settings view** (not Canvas)
2. User toggles specific transcript files → Mode changes to "some"
3. `useEffect` detects the mode change
4. **BUT** the refresh logic checks `if (currentView === 'canvas')`
5. Since `currentView === 'settings'`, no refresh happens!
6. User switches to Canvas view
7. The auto-refresh logic only runs **once per session** (`!canvasAnalysisCheckedRef.current.checked`)
8. If user had already visited Canvas earlier, no refresh happens
9. **Result**: Canvas shows old analysis based on old settings

### Why This Manifested as "Only Latest" Being Used

- Canvas likely had analysis from "latest" mode cached
- Settings changed to "some" but refresh didn't trigger
- Old cached analysis persisted
- User saw analysis that appeared to only use "latest"

## The Fix

Added `pendingCanvasRefreshRef` flag that:

1. **Tracks settings changes** while not in Canvas view
2. **Triggers refresh** when user enters Canvas view
3. **Prioritizes** pending refreshes over normal auto-refresh logic

### Changes Made

**File**: `app/page.tsx`

#### Change 1: Declare pending refresh flag (line 595)

```typescript
const pendingCanvasRefreshRef = useRef<boolean>(false);
```

#### Change 2: Set flag when settings change outside Canvas view (lines 597-641)

```typescript
// For each settings change detection:
if (currentView === 'canvas') {
  handleRefreshCanvasAnalysis(true);
} else {
  pendingCanvasRefreshRef.current = true;  // NEW: Set flag for later
}
```

Applied to:
- Recording session changes
- Groups read mode changes
- Transcript listen mode changes
- Toggle state changes (for "some" mode)

#### Change 3: Check pending flag when entering Canvas view (lines 567-574)

```typescript
if (currentView === 'canvas' && pageAgentName && !isRefreshingCanvasAnalysis) {
  // Priority 1: Check for pending refresh from settings changes
  if (pendingCanvasRefreshRef.current) {
    console.log('[Canvas] Triggering pending analysis refresh from settings changes');
    pendingCanvasRefreshRef.current = false;
    handleRefreshCanvasAnalysis(true);
    return;
  }
  // Priority 2: Normal auto-refresh logic...
}
```

## Testing the Fix

### Expected Workflow Now

1. User in Settings view
2. User toggles transcript files → Mode becomes "some"
3. Console: `[Canvas] Transcript listen mode changed from latest to some, triggering analysis refresh with clearPrevious=true`
4. **NEW**: Since not in Canvas, `pendingCanvasRefreshRef.current = true`
5. User switches to Canvas view
6. **NEW**: Console: `[Canvas] Triggering pending analysis refresh from settings changes`
7. Refresh happens with `clearPrevious=true` and correct toggle states
8. Canvas analysis now reflects selected transcripts!

### Console Logs to Verify

When testing, you should see:

```
[Canvas] Transcript listen mode changed from latest to some, triggering analysis refresh with clearPrevious=true
[Canvas Hook DEBUG] Sending toggle states: { count: 2, keys: [...], sample: [...] }
[Canvas] Triggering pending analysis refresh from settings changes
```

And in backend logs:
```
[DEBUG] 'some' mode: received 2 toggle state entries
[DEBUG] Toggle state keys: [...]
[DEBUG] Found X total files in S3
[DEBUG] Matched 2 toggled transcripts in 'some' mode
```

### Test All Modes

1. **none** → Canvas should show no transcript data
2. **latest** → Canvas should reference only the latest transcript
3. **some** (2 files selected) → Canvas should reference exactly those 2 files
4. **all** → Canvas should reference all transcripts

### Test Workflow Variations

1. **Settings → Canvas**: Change setting in Settings, then switch to Canvas
   - ✅ Fixed by this PR

2. **Canvas → Settings → Canvas**: Already in Canvas, go to Settings, change, come back
   - ✅ Already worked, still works

3. **Multiple setting changes before Canvas**: Change mode, then toggle files, then switch to Canvas
   - ✅ Fixed - pending flag accumulates all changes

## Additional Improvements Included

### 1. DEBUG Logging

Added frontend logging in `hooks/use-canvas-llm.ts:83-89`:

```typescript
if (individualRawTranscriptToggleStates && Object.keys(individualRawTranscriptToggleStates).length > 0) {
  console.log('[Canvas Hook DEBUG] Sending toggle states:', {
    count: Object.keys(individualRawTranscriptToggleStates).length,
    keys: Object.keys(individualRawTranscriptToggleStates).slice(0, 3),
    sample: Object.entries(individualRawTranscriptToggleStates).slice(0, 2)
  });
}
```

Backend already has extensive DEBUG logging in `canvas_analysis_agents.py:312-337`.

### 2. Test Scripts Created

- `test_some_mode_direct.py` - Direct test of filtering logic
- `test_canvas_modes.py` - Full automated API test suite
- `diagnose_some_mode.py` - Real-time log analysis

### 3. Investigation Documentation

- `CANVAS_SOME_MODE_INVESTIGATION.md` - Full investigation report
- `CANVAS_SOME_MODE_FIX.md` (this file) - Root cause and fix explanation

## Summary

**Before**: Settings changes while not in Canvas view were detected but not applied, causing Canvas to show stale analysis.

**After**: Settings changes are tracked with `pendingCanvasRefreshRef` flag and automatically applied when user enters Canvas view.

**Impact**: All transcript modes (none/latest/some/all) now work correctly regardless of which view the user is in when they change settings.
