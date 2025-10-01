-- Migration: Add cross_group_read_enabled to agents table
-- Date: 2025-10-01
-- Description: Moves cross-group read policy from agent_events.event_labels to agents table

-- Step 1: Add the new column to agents table
ALTER TABLE agents
ADD COLUMN IF NOT EXISTS cross_group_read_enabled BOOLEAN DEFAULT FALSE;

-- Step 2: Migrate existing allow_cross_group_read flags from agent_events to agents
-- This finds all agents where event '0000' has allow_cross_group_read=true and sets the agent-level flag
UPDATE agents
SET cross_group_read_enabled = TRUE
WHERE name IN (
    SELECT agent_name
    FROM agent_events
    WHERE event_id = '0000'
    AND event_labels->>'allow_cross_group_read' = 'true'
);

-- Step 3: Create an index for performance
CREATE INDEX IF NOT EXISTS idx_agents_cross_group_read
ON agents(cross_group_read_enabled)
WHERE cross_group_read_enabled = TRUE;

-- Verification query (run separately to check migration):
-- SELECT a.name, a.cross_group_read_enabled, ae.event_labels->>'allow_cross_group_read' as old_flag
-- FROM agents a
-- LEFT JOIN agent_events ae ON a.name = ae.agent_name AND ae.event_id = '0000'
-- WHERE a.cross_group_read_enabled = TRUE OR ae.event_labels->>'allow_cross_group_read' = 'true';
