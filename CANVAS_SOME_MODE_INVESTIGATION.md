# Canvas "Some" Mode Investigation Report

## Problem Statement

User reports that switching transcript selections in Settings > Memory > Transcripts > Listen to "some" mode still only fetches the latest transcript instead of the selected ones.

## Investigation Results

### ✅ VERIFIED: Key Formats Match

Ran direct test (`test_some_mode_direct.py`) and confirmed:
- Frontend sends keys in format: `organizations/river/agents/{agent}/events/{event}/transcripts/{file}.txt`
- Backend expects keys in format: `organizations/river/agents/{agent}/events/{event}/transcripts/{file}.txt`
- **Formats are identical** - no mismatch here

### ✅ IMPLEMENTED: Code Changes

1. **Frontend Toggle State Tracking** (`app/page.tsx:624-632`)
   - Added `previousToggleStatesRef` to detect changes within "some" mode
   - Triggers canvas refresh when toggle states change

2. **Toggle States Passed Through Stack**
   - ✅ `hooks/use-canvas-llm.ts:97` - Sent in canvas stream request
   - ✅ `app/api/canvas/stream/route.ts` - Forwarded to backend
   - ✅ `app/api/canvas/analysis/refresh/route.ts:531` - Sent in refresh request
   - ✅ `routes/canvas_routes.py:180` - Extracted from request
   - ✅ `routes/canvas_routes.py:238` - Passed to analysis function
   - ✅ `utils/canvas_analysis_agents.py:302-337` - Filtered by toggle states

3. **DEBUG Logging Added**
   - Frontend: `hooks/use-canvas-llm.ts:83-89` - Logs toggle states being sent
   - Backend: `utils/canvas_analysis_agents.py:312-337` - Logs toggle states received, S3 keys found, matches

## Possible Root Causes (To Investigate)

Since key formats match and code is wired correctly, the issue might be:

### 1. Empty Toggle States
- Toggle states might be `{}` when sent, even in "some" mode
- **Test**: Check browser console for `[Canvas Hook DEBUG]` logs when refreshing

### 2. Race Condition
- Toggle states might not be updated before refresh is triggered
- **Test**: Add delay between toggle change and refresh

### 3. Mode Not Recognized as "Some"
- Backend might be receiving `transcript_listen_mode=latest` instead of `some`
- **Test**: Check backend logs for "Fetching transcripts for analysis" message

### 4. Supabase Memory Prefs Not Synced
- The transcript_listen_mode in Supabase might not be updating to "some"
- **Test**: Check Supabase `memory_prefs` table for the agent

## Next Steps: Testing with Real Data

### Quick Test (Browser Console)

1. Start the app and open browser DevTools Console
2. Go to Settings > Memory > Transcripts
3. Toggle ON 2 specific transcript files (should enter "some" mode)
4. Switch to Canvas view (should auto-refresh)
5. Look for these console logs:

```
[Canvas] Transcript toggle selection changed in "some" mode, triggering analysis refresh with clearPrevious=true
[Canvas Hook DEBUG] Sending toggle states: { count: 2, keys: [...], sample: [...] }
```

6. Check if the `keys` in the log match the format:
   ```
   organizations/river/agents/{agent}/events/0000/transcripts/{filename}.txt
   ```

### Backend Log Check

While testing, check backend logs (`logs/claude_chat.log`) for:

```bash
tail -f logs/claude_chat.log | grep -E "DEBUG|some mode|toggle"
```

Look for:
```
[DEBUG] 'some' mode: received X toggle state entries
[DEBUG] Toggle state keys: [...]
[DEBUG] Found X total files in S3
[DEBUG] S3 file keys: [...]
[DEBUG] Matched X toggled transcripts
[DEBUG] Matched keys: [...]
```

**If you see `[DEBUG] NO MATCHES!`** - this is the smoking gun!

### Full Automated Test

Once you have a real agent with transcript files, run:

```bash
# Edit the agent name in the script first
python3 test_some_mode_direct.py --agent YOUR_AGENT_NAME
```

This will:
1. Verify key formats match
2. List actual transcript files
3. Create toggle states for first 2 files
4. Test the filtering logic
5. Report success/failure with details

## Diagnostic Scripts

### 1. `test_some_mode_direct.py`
Direct test of filtering logic with mock toggle states

### 2. `diagnose_some_mode.py`
Real-time log analysis tool

```bash
# Watch logs while testing in UI
python3 diagnose_some_mode.py --watch
```

### 3. `test_canvas_modes.py`
Full automated API test suite (requires auth token)

```bash
# Get token from browser DevTools > Application > Cookies > sb-access-token
python3 test_canvas_modes.py --agent YOUR_AGENT --token YOUR_TOKEN --loop
```

## Summary

**Conclusion**: Code is correctly wired, key formats match. The issue is likely one of:
1. Toggle states not being populated
2. Timing/race condition
3. Supabase memory_prefs not updating
4. Mode detection logic issue

**Recommendation**: Run the browser console test above to see what's actually being sent. The DEBUG logs will reveal the issue.
