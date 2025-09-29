-- CORRECTED Database Constraint for Agent Names
-- This version allows legitimate patterns while blocking the problematic ones

-- Drop the overly-restrictive constraint if it was added
ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_name_format;

-- Add a more permissive constraint that only blocks the specific problematic patterns:
-- 1. Blocks wizard-session-* patterns
-- 2. Blocks pure 32-character hex UUIDs (like e5d8bea81b8c49d7a9612e1dc97c09de)
-- 3. Allows underscores, mixed case, numbers at start, etc. for legitimate agents

ALTER TABLE agents
ADD CONSTRAINT agents_name_format
CHECK (
    -- Block wizard session IDs
    NOT (name LIKE 'wizard-session%')
    AND
    -- Block pure 32-char hex UUIDs (but allow names with UUIDs as part of them)
    NOT (name ~ '^[0-9a-f]{32}$')
);

-- Test the constraint with your legitimate agent names:
-- These should all PASS:
-- ✓ bricks_control_a-raw (has underscore and hyphen)
-- ✓ bricks_control_b-context
-- ✓ _test (starts with underscore)
-- ✓ _agent (starts with underscore)
-- ✓ 03events (starts with number)
-- ✓ 00gradient (starts with number)

-- These should all FAIL:
-- ✗ wizard-session-abc123def456
-- ✗ e5d8bea81b8c49d7a9612e1dc97c09de (32 hex chars)