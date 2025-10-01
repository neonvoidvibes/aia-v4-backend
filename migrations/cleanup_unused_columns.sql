-- Cleanup Migration: Remove Unused Columns
-- Date: 2025-10-01
-- Description: Removes obsolete columns after cross_group_read migration

-- ============================================================================
-- ANALYSIS SUMMARY
-- ============================================================================
--
-- 1. agent_events.event_labels (JSONB)
--    - Previously stored: { "allow_cross_group_read": true/false }
--    - Status: MIGRATED to agents.cross_group_read_enabled
--    - Current usage: Loaded but never accessed in code (lines 1489-1494)
--    - Safe to remove: YES ✅
--
-- 2. agents.cross_event_access (unknown type)
--    - Status: Leftover column, no references found in codebase
--    - Safe to remove: YES ✅
--
-- ============================================================================

-- STEP 1: Verify no usage in application code
-- Run these queries to confirm data distribution before dropping:

-- Check event_labels contents across all events
-- SELECT agent_name, event_id, event_labels
-- FROM agent_events
-- WHERE event_labels IS NOT NULL AND event_labels::text != '{}'
-- ORDER BY agent_name, event_id;

-- Check cross_event_access contents
-- SELECT name, cross_event_access
-- FROM agents
-- WHERE cross_event_access IS NOT NULL;

-- ============================================================================
-- STEP 2: Backup existing data (RECOMMENDED)
-- ============================================================================

-- Backup event_labels to a temporary table (optional, for rollback)
CREATE TABLE IF NOT EXISTS agent_events_event_labels_backup AS
SELECT agent_name, event_id, event_labels, updated_at
FROM agent_events
WHERE event_labels IS NOT NULL AND event_labels::text != '{}';

-- Backup cross_event_access to a temporary table (optional, for rollback)
CREATE TABLE IF NOT EXISTS agents_cross_event_access_backup AS
SELECT name, cross_event_access, updated_at
FROM agents
WHERE cross_event_access IS NOT NULL;

-- ============================================================================
-- STEP 3: Drop unused columns
-- ============================================================================

-- Drop event_labels from agent_events
ALTER TABLE agent_events DROP COLUMN IF EXISTS event_labels;

-- Drop cross_event_access from agents
ALTER TABLE agents DROP COLUMN IF EXISTS cross_event_access;

-- ============================================================================
-- VERIFICATION QUERIES (run after migration)
-- ============================================================================

-- Verify columns are removed
-- \d agent_events  -- should NOT show event_labels
-- \d agents        -- should NOT show cross_event_access

-- Verify backup tables exist
-- SELECT COUNT(*) FROM agent_events_event_labels_backup;
-- SELECT COUNT(*) FROM agents_cross_event_access_backup;

-- ============================================================================
-- ROLLBACK (if needed)
-- ============================================================================

-- Restore event_labels
-- ALTER TABLE agent_events ADD COLUMN event_labels JSONB;
-- UPDATE agent_events ae
-- SET event_labels = b.event_labels
-- FROM agent_events_event_labels_backup b
-- WHERE ae.agent_name = b.agent_name AND ae.event_id = b.event_id;

-- Restore cross_event_access
-- ALTER TABLE agents ADD COLUMN cross_event_access <TYPE>;  -- replace <TYPE> with actual type
-- UPDATE agents a
-- SET cross_event_access = b.cross_event_access
-- FROM agents_cross_event_access_backup b
-- WHERE a.name = b.name;

-- Drop backup tables (only after verifying migration success)
-- DROP TABLE IF EXISTS agent_events_event_labels_backup;
-- DROP TABLE IF EXISTS agents_cross_event_access_backup;

-- ============================================================================
-- NOTES
-- ============================================================================
--
-- 1. The event_labels column was only used to store allow_cross_group_read,
--    which has been migrated to agents.cross_group_read_enabled.
--
-- 2. The code still loads event_labels (api_server.py:1489-1494) but never
--    accesses it. After this migration, remove those lines too:
--    ```python
--    # DELETE THESE LINES (api_server.py:1489-1494)
--    event_labels_data = row.get("event_labels") or {}
--    if isinstance(event_labels_data, str):
--        try:
--            event_labels_data = json.loads(event_labels_data)
--        except Exception:
--            event_labels_data = {}
--    ```
--
-- 3. Also remove event_labels from the SELECT query (api_server.py:1400):
--    ```python
--    # CHANGE FROM:
--    "event_id,type,visibility_hidden,owner_user_id,event_labels,workspace_id,created_at"
--    # TO:
--    "event_id,type,visibility_hidden,owner_user_id,workspace_id,created_at"
--    ```
--
-- 4. cross_event_access has no references in the codebase, safe to drop.
--
-- ============================================================================
