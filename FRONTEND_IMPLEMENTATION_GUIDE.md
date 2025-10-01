# Frontend Implementation Guide: Cross-Group Read Toggle

## Quick Summary
This guide provides exact code changes to implement the cross-group read toggle in the frontend.

---

## Step 1: Add State Variables (`app/page.tsx`)

### Location: Near line 411 (with other state declarations)

```typescript
// Add these new state variables
const [allowCrossGroupRead, setAllowCrossGroupRead] = useState<boolean>(false);
const [allowedGroupEventsCount, setAllowedGroupEventsCount] = useState<number>(0);
```

---

## Step 2: Update Event Fetching Logic

### Location: Around line 595 (where events are fetched)

**Current code** (approximately line 593-596):
```typescript
const res = await fetch(`/api/s3-proxy/list-events?agentName=${encodeURIComponent(pageAgentName)}`);
if (res.ok) {
  const data: { events?: string[]; eventTypes?: Record<string, string>; allowedEvents?: string[]; personalEventId?: string | null } = await res.json();
```

**Replace with**:
```typescript
const res = await fetch(`/api/s3-proxy/list-events?agentName=${encodeURIComponent(pageAgentName)}`);
if (res.ok) {
  const data: {
    events?: string[];
    eventTypes?: Record<string, string>;
    allowedEvents?: string[];
    personalEventId?: string | null;
    allowCrossGroupRead?: boolean;          // NEW
    allowedGroupEventsCount?: number;      // NEW
  } = await res.json();

  // Store cross-group read state
  setAllowCrossGroupRead(data.allowCrossGroupRead || false);
  setAllowedGroupEventsCount(data.allowedGroupEventsCount || 0);
```

---

## Step 3: Add LocalStorage Utilities

### Create new file: `lib/crossGroupReadStorage.ts`

```typescript
/**
 * LocalStorage utilities for cross-group read preference
 */

export const getCrossGroupReadKey = (userId: string, agentName: string): string => {
  return `crossGroupRead:${userId}:${agentName}`;
};

export const loadCrossGroupReadPreference = (userId: string, agentName: string): boolean => {
  if (typeof window === 'undefined') return false;
  const key = getCrossGroupReadKey(userId, agentName);
  const value = localStorage.getItem(key);
  return value === 'true';
};

export const saveCrossGroupReadPreference = (userId: string, agentName: string, enabled: boolean): void => {
  if (typeof window === 'undefined') return;
  const key = getCrossGroupReadKey(userId, agentName);
  localStorage.setItem(key, String(enabled));
};
```

---

## Step 4: Load Preference on Agent Change

### Location: In the effect that runs when pageAgentName or pageEventId changes

**Add this near the event fetching effect** (after line 670):

```typescript
// Load cross-group read preference from localStorage
useEffect(() => {
  if (!pageAgentName || !userId) return;

  const savedPref = loadCrossGroupReadPreference(userId, pageAgentName);
  setAllowCrossGroupRead(savedPref);
}, [pageAgentName, userId]);
```

---

## Step 5: Add API Call Function

### Location: Near other API handlers (around line 800-1000)

```typescript
// Handler for updating cross-group read setting
const handleCrossGroupReadToggle = async (enabled: boolean) => {
  if (!pageAgentName || !pageEventId) return;

  // Update UI immediately
  setAllowCrossGroupRead(enabled);

  // Save to localStorage
  if (userId) {
    saveCrossGroupReadPreference(userId, pageAgentName, enabled);
  }

  // Only sync to backend if we're in event 0000
  if (pageEventId !== '0000') return;

  try {
    const response = await fetch(`/api/agent-events/${pageAgentName}/${pageEventId}/labels`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        event_labels: {
          allow_cross_group_read: enabled
        }
      })
    });

    if (!response.ok) {
      console.error('Failed to update cross-group read setting');
      // Optionally show toast notification
    }
  } catch (error) {
    console.error('Error updating cross-group read setting:', error);
    // Optionally show toast notification
  }
};
```

---

## Step 6: Add Toggle to Settings UI

### Location: Inside the "Transcripts" CollapsibleSection (after line 2929)

**Insert this code RIGHT AFTER the Listen toggle group closes (after line 2929)**:

```tsx
{/* NEW: Cross-group read toggle */}
<div className="flex items-center justify-between py-3 border-b mb-3">
  <div className="flex flex-col gap-1">
    <div className="flex items-center gap-2">
      <Users className="h-5 w-5 text-muted-foreground" />
      <Label
        htmlFor="cross-group-read-toggle"
        className="memory-section-title text-sm font-medium cursor-pointer"
      >
        Read other groups' transcripts
      </Label>
    </div>
    <p className="text-xs text-muted-foreground ml-7">
      {pageEventId === '0000'
        ? "Enables reading live transcripts from all group events simultaneously"
        : "Only available in event 0000"}
    </p>
  </div>
  <Switch
    id="cross-group-read-toggle"
    checked={allowCrossGroupRead}
    disabled={pageEventId !== '0000'}
    onCheckedChange={handleCrossGroupReadToggle}
  />
</div>
```

**Add the Users import at the top of the file**:
```typescript
import { Users } from 'lucide-react'; // Add to existing lucide-react imports
```

---

## Step 7: Update Status Text

### Location: In the `simple-chat-interface.tsx` component

**Find the code around line 1152-1184** that derives the listening mode status text.

**Current code** (approximately):
```typescript
// Derive listening mode and optional +N based on Settings > Memory selections
const listeningModeText = useMemo(() => {
  // ... existing logic
  if (transcriptListenMode === 'latest') {
    return "Listening to latest";
  } else if (transcriptListenMode === 'some') {
    // ...
  }
  // ... rest of logic
}, [transcriptListenMode, ...]);
```

**Modify to**:

1. **First, add props to the component interface** (around line 448):

```typescript
interface SimpleChatInterfaceProps {
  // ... existing props
  transcriptListenMode: "none" | "some" | "latest" | "all";
  allowCrossGroupRead?: boolean;            // NEW
  allowedGroupEventsCount?: number;         // NEW
  // ... rest of props
}
```

2. **Then update the status text logic** (around line 1159-1184):

```typescript
const listeningModeText = useMemo(() => {
  const isCrossGroup = eventId === '0000' && allowCrossGroupRead && (allowedGroupEventsCount || 0) > 0;
  const groupsSuffix = isCrossGroup ? ` groups +${allowedGroupEventsCount}` : '';

  // Transcript listen mode
  if (transcriptListenMode === 'latest') {
    return `Listening to latest${groupsSuffix}`;
  } else if (transcriptListenMode === 'some') {
    const activeRawCount = Object.values(individualRawTranscriptToggleStates || {}).filter(Boolean).length;
    if (activeRawCount === 1) {
      return isCrossGroup ? `Listening to groups +${allowedGroupEventsCount}` : `Listening to 1 transcript`;
    }
    return isCrossGroup ? `Listening to groups +${allowedGroupEventsCount}` : `Listening to ${activeRawCount} transcripts`;
  } else if (transcriptListenMode === 'all') {
    return isCrossGroup ? `Listening to groups +${allowedGroupEventsCount}` : `Listening to all transcripts`;
  }

  // ... rest of the existing logic for savedTranscriptMemoryMode
}, [
  transcriptListenMode,
  individualRawTranscriptToggleStates,
  rawTranscriptFiles,
  savedTranscriptMemoryMode,
  individualMemoryToggleStates,
  savedTranscriptSummaries,
  eventId,                     // NEW
  allowCrossGroupRead,         // NEW
  allowedGroupEventsCount      // NEW
]);
```

3. **Pass the new props when rendering SimpleChatInterface** in `app/page.tsx` (around line 2563):

```tsx
<SimpleChatInterface
  // ... existing props
  transcriptListenMode={transcriptListenMode}
  allowCrossGroupRead={allowCrossGroupRead}           // NEW
  allowedGroupEventsCount={allowedGroupEventsCount}   // NEW
  // ... rest of props
/>
```

---

## Step 8: Update Backend Proxy (if needed)

### Check if `/api/s3-proxy/list-events` needs updating

The backend already returns `allowCrossGroupRead` and `allowedGroupEventsCount` in the `/api/s3/list-events` endpoint.

If there's a Next.js API proxy route, ensure it forwards these fields:

### File: `app/api/s3-proxy/list-events/route.ts` (or similar)

```typescript
// Ensure the proxy forwards all fields from the backend response
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const agentName = searchParams.get('agentName');

  const backendRes = await fetch(
    `${process.env.BACKEND_URL}/api/s3/list-events?agent=${agentName}`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
      }
    }
  );

  const data = await backendRes.json();

  // Return all fields including the new ones
  return Response.json(data);
}
```

---

## Testing Checklist

After implementing the above changes:

### UI Tests
- [ ] Toggle appears in Settings > Memory > Transcripts
- [ ] Toggle is grayed out when not in event 0000
- [ ] Toggle label shows "Only available in event 0000" tooltip when disabled
- [ ] Toggle can be switched ON/OFF when in event 0000

### Persistence Tests
- [ ] Toggle state persists after page reload
- [ ] Toggle state persists when switching agents (per-agent)
- [ ] Toggle state persists when switching events

### Backend Integration Tests
- [ ] PATCH request is sent to backend when toggle changes (check Network tab)
- [ ] Backend returns success response
- [ ] Supabase `event_labels.allow_cross_group_read` is updated

### Status Text Tests
- [ ] Status shows "Listening to latest" when toggle OFF
- [ ] Status shows "Listening to latest groups +2" when toggle ON (with 2 group events)
- [ ] Status updates immediately when toggle changes
- [ ] Group count is accurate

### Functional Tests
- [ ] Chat request includes correct transcript data
- [ ] Backend logs show: "Cross-group read: fetching latest transcripts from N events"
- [ ] LLM receives transcripts from multiple events with labels

---

## File Summary

Files to modify:
1. `app/page.tsx` - Add state, toggle UI, handlers
2. `components/simple-chat-interface.tsx` - Update status text logic
3. `lib/crossGroupReadStorage.ts` - NEW file for localStorage utilities
4. Import `Users` icon from lucide-react

---

## Quick Diff Summary

### `app/page.tsx`
```diff
+ const [allowCrossGroupRead, setAllowCrossGroupRead] = useState<boolean>(false);
+ const [allowedGroupEventsCount, setAllowedGroupEventsCount] = useState<number>(0);

  const data: {
    events?: string[];
    eventTypes?: Record<string, string>;
+   allowCrossGroupRead?: boolean;
+   allowedGroupEventsCount?: number;
  } = await res.json();

+ setAllowCrossGroupRead(data.allowCrossGroupRead || false);
+ setAllowedGroupEventsCount(data.allowedGroupEventsCount || 0);

+ const handleCrossGroupReadToggle = async (enabled: boolean) => { /* ... */ };

  <SimpleChatInterface
    transcriptListenMode={transcriptListenMode}
+   allowCrossGroupRead={allowCrossGroupRead}
+   allowedGroupEventsCount={allowedGroupEventsCount}
  />

+ {/* Cross-group read toggle in Transcripts section */}
+ <div className="flex items-center justify-between py-3 border-b mb-3">
+   <Switch id="cross-group-read-toggle" checked={allowCrossGroupRead} ... />
+ </div>
```

### `components/simple-chat-interface.tsx`
```diff
interface SimpleChatInterfaceProps {
+ allowCrossGroupRead?: boolean;
+ allowedGroupEventsCount?: number;
}

const listeningModeText = useMemo(() => {
+ const isCrossGroup = eventId === '0000' && allowCrossGroupRead && (allowedGroupEventsCount || 0) > 0;
+ const groupsSuffix = isCrossGroup ? ` groups +${allowedGroupEventsCount}` : '';

  if (transcriptListenMode === 'latest') {
-   return "Listening to latest";
+   return `Listening to latest${groupsSuffix}`;
  }
}, [..., eventId, allowCrossGroupRead, allowedGroupEventsCount]);
```

---

## Estimated Effort
- Implementation: 2-3 hours
- Testing: 1 hour
- Total: 3-4 hours

---

## Support
If you encounter issues:
1. Check browser console for errors
2. Check Network tab for API calls
3. Check backend logs for "Cross-group read" messages
4. Verify localStorage keys in Application tab

All backend endpoints are ready and working. This is purely frontend integration.
