# Metadata Migration for Enhanced Vector Retrieval

This documentation describes the metadata migration process for updating existing Pinecone vectors with enhanced metadata fields to support improved content categorization and retrieval.

## Overview

The migration system intelligently categorizes existing content and adds structured metadata fields that enable:
- **Content categorization**: meeting_summary, chat_message, foundational_document, other
- **Temporal relevance**: time_sensitive, persistent, contextual  
- **Analysis type detection**: multi_agent, single_layer, summary
- **Date extraction**: Automatic extraction from filenames and metadata

## Files

- `scripts/migrate_metadata.py` - Main migration script
- `scripts/run_migration.sh` - User-friendly wrapper script
- `scripts/MIGRATION_README.md` - This documentation

## Quick Start

### 1. Preview Migration (Safe)
```bash
# Preview changes for specific event
./scripts/run_migration.sh --dry-run --event 0000

# Preview changes for all events  
./scripts/run_migration.sh --dry-run --event all
```

### 2. Execute Migration (DANGER!)
```bash
# Execute migration for specific event
./scripts/run_migration.sh --execute --event 0000

# Execute migration for all events
./scripts/run_migration.sh --execute --event all --batch-size 50
```

## Content Detection Logic

### Meeting Summaries
**Detected by:**
- Filename contains: `transcript_`, `summary_`, `_full.md`, `_business_reality.md`, `_context.md`
- Content keywords: `meeting`, `workshop`

**Enhanced metadata:**
```json
{
  "content_category": "meeting_summary",
  "analysis_type": "multi_agent|single_layer|summary",
  "temporal_relevance": "time_sensitive", 
  "meeting_date": "2025-08-12",
  "summary_type": "comprehensive|business_focused|contextual|standard"
}
```

### Chat Messages  
**Detected by:**
- Existing metadata fields: `message_id`, `user_id`, `timestamp`, `chat_id`
- Vector ID contains: `chat`

**Enhanced metadata:**
```json
{
  "content_category": "chat_message",
  "temporal_relevance": "time_sensitive",
  "interaction_type": "user_message|assistant_message|system_message",
  "message_date": "2025-08-12"
}
```

### Foundational Documents
**Detected by:**
- Filename contains: `systemprompt`, `context`, `instructions`, `guidelines`, `framework`, `template`, `config`, `strategy`

**Enhanced metadata:**
```json
{
  "content_category": "foundational_document", 
  "temporal_relevance": "persistent",
  "doc_type": "system_prompt|context_document|instructions|framework|reference",
  "scope": "global|event_specific"
}
```

### Other Content
**All remaining content:**
```json
{
  "content_category": "other",
  "temporal_relevance": "contextual"
}
```

## Safety Features

### Dry Run Mode (Default)
- **Always runs in dry-run mode by default**
- Previews all changes without modifying data
- Generates detailed statistics
- Safe to run multiple times

### Execution Mode Safeguards
- Requires explicit `--execute` flag
- Interactive confirmation prompt
- Batch processing with error handling
- Skip vectors that already have new metadata
- Comprehensive logging and error tracking

### Data Protection
- **No data loss**: Only adds/updates metadata, never removes existing fields
- **Idempotent**: Safe to run multiple times - skips already migrated vectors
- **Rollback**: Existing metadata is preserved and can be restored if needed

## Batch Processing

The migration processes vectors in configurable batches:
- Default batch size: 100 vectors
- Adjustable with `--batch-size` parameter
- Progress tracking with batch numbers
- Individual error handling per batch

## Statistics and Monitoring

### Real-time Stats
```
Migration completed!
Total processed: 1,234
Meeting summaries: 456
Chat messages: 678
Foundational docs: 89
Other content: 11
Updated: 1,220
Skipped (already migrated): 14
Errors: 0
```

### Stats File
Each migration run creates a timestamped JSON stats file:
```
migration_stats_river_20250911_215344.json
```

## Advanced Usage

### Direct Python Script
```bash
# Specific event with custom batch size
python scripts/migrate_metadata.py \
  --index-name river \
  --event-id 0000 \
  --dry-run \
  --batch-size 50 \
  --log-level DEBUG

# Execute migration for all events
python scripts/migrate_metadata.py \
  --index-name river \
  --event-id all \
  --execute \
  --batch-size 100
```

### Command Line Options
```
--index-name REQUIRED    Pinecone index name
--event-id               Event ID filter ('all' for all events) 
--dry-run               Preview mode (safe)
--execute               Execution mode (DANGER!)
--batch-size INT        Vectors per batch (default: 100)
--log-level             DEBUG|INFO|WARNING|ERROR
```

## Migration Strategy Recommendations

### For IKEA Pilot Context

1. **Test with Single Event First**
   ```bash
   ./scripts/run_migration.sh --dry-run --event 0000
   ```

2. **Migrate Recent Meeting Summaries**
   ```bash
   # Focus on events with meeting data
   ./scripts/run_migration.sh --execute --event cookingandeating
   ./scripts/run_migration.sh --execute --event folkhemmen
   ```

3. **Full Migration (Production)**
   ```bash
   ./scripts/run_migration.sh --execute --event all --batch-size 50
   ```

### Performance Considerations

- **Small batches** (10-50) for initial testing
- **Medium batches** (100-200) for production migration  
- **Monitor logs** for any errors or timeouts
- **Run during low-usage periods** to avoid API rate limits

## Troubleshooting

### Common Issues

**No vectors found to migrate**
- Verify index name is correct
- Check if event_id filter is too restrictive
- Use `--event all` to see all available vectors

**Import errors**
- Ensure you're running from the backend root directory
- Check that all dependencies are installed
- Verify Python path includes utils modules

**API rate limits**
- Reduce batch size with `--batch-size 10`
- Add delays between batches (future enhancement)

**Migration failures**
- Check migration stats file for detailed error information
- Re-run with `--log-level DEBUG` for detailed logging
- Vectors with errors are marked but migration continues

### Recovery

If migration encounters issues:
1. Check the stats file for specific error details
2. Re-run migration - it will skip already-migrated vectors
3. Use smaller batch sizes to isolate problematic vectors
4. Contact system administrator if persistent errors occur

## Integration with Enhanced Retrieval

After migration, the enhanced retrieval system in `api_server.py` will automatically use the new metadata fields:

```python
metadata_filter = {
    "content_category": "meeting_summary",
    "event_id": event_id,
    "analysis_type": "multi_agent"
}
```

This enables precise filtering and improved context retrieval for the IKEA pilot scenarios.