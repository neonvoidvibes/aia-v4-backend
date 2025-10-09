-- Migration: Add 'breakout' to groups_read_mode constraint
-- Date: 2025-10-09
-- Purpose: Allow 'breakout' as a valid value for groups_read_mode

-- Drop the existing constraint
ALTER TABLE agents
DROP CONSTRAINT IF EXISTS groups_read_mode_check;

-- Add new constraint with 'breakout' included
ALTER TABLE agents
ADD CONSTRAINT groups_read_mode_check CHECK (groups_read_mode IN ('latest', 'none', 'all', 'breakout'));
