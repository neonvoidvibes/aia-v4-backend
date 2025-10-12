## Manual Test Procedure for Canvas "Some" Mode

### CRITICAL ISSUE IDENTIFIED

The backend reads `transcript_listen_mode` from **Supabase agents table** (canvas_routes.py:194), not from toggle states!

**This is the problem**: When you toggle files in the UI, the mode needs to be saved to Supabase, but the backend only uses toggle states IF the mode is "some".

### Test Procedure (Loop Until Working)

#### Terminal 1: Watch Backend Logs

```bash
cd /Users/neonvoid/Documents/AI_apps/aia-v4/aia-v4-backend
bash watch_canvas_test.sh
```

This will show real-time logs filtered for canvas analysis activity.

#### Terminal 2: UI Testing

1. **Start the app** (frontend + backend running)

2. **Open browser DevTools Console** (F12 → Console tab)

3. **Test Scenario: Latest → Some Transition**
   ```
   a. Go to Settings > Memory > Transcripts
   b. Set Listen to "latest"
   c. Wait 1 second
   d. Toggle ON 2-3 specific transcript files
   e. Mode should auto-switch to "some"
   f. Switch to Canvas view
   ```

4. **Watch for these logs in Terminal 1:**
   ```
   ✅ EXPECTED:
   Canvas: transcript_listen_mode=some, groups_read_mode=none for {agent}
   [DEBUG] 'some' mode: received 2-3 toggle state entries
   [DEBUG] Matched 2-3 toggled transcripts

   ❌ IF YOU SEE THIS:
   Canvas: transcript_listen_mode=latest, groups_read_mode=none for {agent}
   → Problem: Supabase not updating to 'some' mode
   ```

5. **Check Browser Console for:**
   ```
   [Canvas] Transcript toggle selection changed in "some" mode
   [Canvas Hook DEBUG] Sending toggle states: { count: 2-3, keys: [...] }
   ```

#### What to Check If It Fails

##### Issue 1: Supabase Not Updating

**Check in browser console:**
```javascript
// After toggling files, check:
localStorage.getItem('transcriptListenModeSetting_{agent}_{userId}')
// Should show "some"
```

**Verify API call:**
- Open DevTools → Network tab
- Filter for "memory-prefs"
- When you toggle files, you should see POST to `/api/agents/memory-prefs`
- Check payload: should have `transcript_listen_mode: "some"`
- Check response: should be 200 OK

**If API call is missing/failing:**
→ Problem in `handleIndividualRawTranscriptToggleChange` or `handleTranscriptListenModeChange`

##### Issue 2: Toggle States Not Sent

**Check in browser console after switching to Canvas:**
```
[Canvas Hook DEBUG] Sending toggle states: ...
```

**If this log is missing:**
→ `individualRawTranscriptToggleStates` is empty
→ Check `handleIndividualRawTranscriptToggleChange` in page.tsx:1455

**If keys in log don't match S3 keys:**
→ Key format mismatch (but we verified formats match)

##### Issue 3: Backend Ignoring Toggle States

**Check in Terminal 1 logs:**
```
[DEBUG] 'some' mode: received X toggle state entries
[DEBUG] Toggle state keys: [...]
[DEBUG] Found X total files in S3
[DEBUG] S3 file keys: [...]
[DEBUG] Matched X toggled transcripts
```

**If "received 0 toggle state entries":**
→ Frontend not sending toggle states
→ Check `handleRefreshCanvasAnalysis` at page.tsx:531

**If "Matched 0 toggled transcripts":**
→ Key mismatch between frontend keys and S3 keys
→ Print both and compare formats

### Loop Until Fixed

Keep testing the scenario above and checking logs. After each failed attempt:

1. Identify which issue from above
2. Fix the code
3. Restart frontend (if frontend changes)
4. Test again
5. Repeat until you see:
   ```
   ✅ Canvas: transcript_listen_mode=some
   ✅ [DEBUG] 'some' mode: received 2-3 toggle state entries
   ✅ [DEBUG] Matched 2-3 toggled transcripts
   ✅ Canvas analysis references exactly those 2-3 files
   ```

### Quick Verification Commands

**Check Supabase directly:**
```sql
-- Run in Supabase SQL Editor
SELECT name, transcript_listen_mode, groups_read_mode
FROM agents
WHERE name = '{your_agent_name}';
```

**Check what files exist in S3:**
```bash
# If AWS CLI is configured
aws s3 ls s3://your-bucket/organizations/river/agents/{agent}/events/0000/transcripts/
```

**Check latest backend log entries:**
```bash
tail -50 logs/claude_chat.log | grep -E "Canvas:|DEBUG"
```

### Success Criteria

Test passes when ALL of these are true:

1. ✅ Frontend console shows: `Sending toggle states: { count: 2-3, ... }`
2. ✅ Backend log shows: `transcript_listen_mode=some`
3. ✅ Backend log shows: `received 2-3 toggle state entries`
4. ✅ Backend log shows: `Matched 2-3 toggled transcripts`
5. ✅ Canvas analysis document references exactly those 2-3 files (not all, not latest only)

### Most Likely Issue

Based on the code review, the most likely issue is:

**Supabase agents table has `transcript_listen_mode=latest` instead of `some`**

This happens if:
- The frontend isn't calling the API to update Supabase
- The API call is failing silently
- There's a race condition between mode change and canvas refresh

**To verify**, check Network tab for POST to `/api/agents/memory-prefs` when you toggle files.
