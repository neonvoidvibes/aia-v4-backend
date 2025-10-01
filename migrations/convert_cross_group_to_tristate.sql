-- Migration: Convert cross_group_read_enabled (boolean) to groups_read_mode (text: 'latest'|'none'|'all')
-- Date: 2025-10-01

-- Step 1: Add new column
ALTER TABLE agents
ADD COLUMN IF NOT EXISTS groups_read_mode TEXT DEFAULT 'none';

-- Step 2: Migrate existing data (true -> 'latest', false -> 'none')
UPDATE agents
SET groups_read_mode = CASE
    WHEN cross_group_read_enabled = TRUE THEN 'latest'
    ELSE 'none'
END;

-- Step 3: Add check constraint to ensure valid values
ALTER TABLE agents
ADD CONSTRAINT groups_read_mode_check CHECK (groups_read_mode IN ('latest', 'none', 'all'));

-- Step 4: Drop old column (commented out for safety - uncomment after verification)
-- ALTER TABLE agents DROP COLUMN cross_group_read_enabled;

-- Note: Keep both columns temporarily to allow rollback if needed
-- After verification, run: ALTER TABLE agents DROP COLUMN cross_group_read_enabled;
