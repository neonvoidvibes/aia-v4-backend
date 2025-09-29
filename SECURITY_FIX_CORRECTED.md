# Security Fix - Corrected Version
## Prevent Wizard Session Agent Auto-Creation

**Date:** 2025-01-29
**Status:** CORRECTED - More permissive validation

---

## Summary of Changes

After discovering legitimate agents like `_agent`, `_test`, `03events`, and `bricks_control_a-raw`, the validation was **corrected** to be more permissive while still blocking the problematic patterns.

---

## ✅ What Gets BLOCKED

1. **Wizard session IDs**: `wizard-session-{anything}`
2. **Pure 32-character hex UUIDs**: `e5d8bea81b8c49d7a9612e1dc97c09de`

## ✅ What Gets ALLOWED

- Underscores: `_agent`, `_test`, `bricks_control_a-raw`
- Starting with numbers: `03events`, `00gradient`, `01organizations`, `02agents`
- Mixed case: `MyAgent` (if needed)
- Hyphens: `customer-support`, `test-bot-2`
- Any combination of alphanumeric, underscore, hyphen

---

## Implementation Details

### Fix 1: Agent Creation Validation
**Location:** `api_server.py:4146-4156`

```python
# SECURITY: Validate agent name format to prevent random UUID/session ID abuse
# Allow flexible naming (underscores, mixed case, numbers, etc.) but block specific problematic patterns
if agent_name.startswith('wizard-session'):
    logger.warning(f"BLOCKED: Wizard session ID attempted as agent name: '{agent_name}' by user {user.id}")
    return jsonify({"error": "Invalid agent name. Cannot use wizard session IDs."}), 400

if re.match(r'^[0-9a-f]{32}$', agent_name):
    logger.warning(f"BLOCKED: Pure UUID pattern attempted as agent name: '{agent_name}' by user {user.id}")
    return jsonify({"error": "Invalid agent name. Cannot use random UUID patterns."}), 400
```

### Fix 2: Chat History Save Protection
**Location:** `api_server.py:5473-5501`

```python
# SECURITY: Block wizard sessions from creating chat history/agent entries
if client_session_id and client_session_id.startswith('wizard-session'):
    logger.info(f"BLOCKED: Wizard session {client_session_id}...")
    return jsonify({'success': True, 'chatId': None, 'message': 'Wizard sessions are not saved'}), 200

# Additional validation: prevent UUID-like or wizard-session agent names
if agent_name.startswith('wizard-session') or re.match(r'^[0-9a-f]{32}$', agent_name):
    logger.warning(f"BLOCKED: Suspicious agent name in chat save...")
    return jsonify({'success': True, 'chatId': None, 'message': 'Invalid agent name pattern'}), 200
```

### Fix 3: S3 Sync Validation
**Location:** `api_server.py:5052-5056`

```python
# SECURITY: Filter out invalid agent names (wizard sessions, UUIDs, etc.)
# Allow flexible naming but block specific problematic patterns
def is_valid_agent_name(name):
    return (not name.startswith('wizard-session') and
            not re.match(r'^[0-9a-f]{32}$', name))
```

---

## Database Constraint (Priority 3)

Run this SQL to add the corrected constraint:

```sql
-- Remove old constraint if present
ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_name_format;

-- Add corrected constraint
ALTER TABLE agents
ADD CONSTRAINT agents_name_format
CHECK (
    NOT (name LIKE 'wizard-session%')
    AND NOT (name ~ '^[0-9a-f]{32}$')
);
```

---

## Testing

### ✅ Should PASS (Legitimate agents)
- `bricks_control_a-raw`
- `bricks_control_b-context`
- `_test`
- `_agent`
- `03events`
- `00gradient`
- `01organizations`
- `customer-support-bot`

### ❌ Should FAIL (Problematic patterns)
- `wizard-session-abc123`
- `wizard-session-e5d8bea81b8c49d7a9612e1dc97c09de`
- `e5d8bea81b8c49d7a9612e1dc97c09de` (pure 32-char hex)
- `a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4` (pure 32-char hex)

---

## Validation

```bash
✓ python -m py_compile api_server.py  # Passed
```

---

## Summary

The validation is now **minimally restrictive**:
- Only blocks the two specific problematic patterns
- Allows all legitimate naming conventions
- Maintains security against wizard session auto-creation
- Provides comprehensive logging for audit trail

**Status:** ✅ Ready for deployment