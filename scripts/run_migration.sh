#!/bin/bash

# Metadata Migration Helper Script
# This script provides safe wrappers for running the metadata migration

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INDEX_NAME="river"
EVENT_ID=""
DRY_RUN=true
BATCH_SIZE=100

show_help() {
    echo "Metadata Migration Helper"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -i, --index INDEX_NAME    Pinecone index name (default: river)"
    echo "  -e, --event EVENT_ID      Event ID to migrate (or 'all')"
    echo "  -n, --dry-run            Preview changes without executing (default)"
    echo "  -x, --execute            Execute the migration (DANGER!)"
    echo "  -b, --batch-size SIZE    Batch size (default: 100)"
    echo "  -h, --help               Show this help"
    echo
    echo "Examples:"
    echo "  $0 --dry-run --event 0000                    # Preview migration for event 0000"
    echo "  $0 --execute --event 0000 --batch-size 50    # Execute migration for event 0000"
    echo "  $0 --dry-run --event all                     # Preview migration for all events"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--index)
            INDEX_NAME="$2"
            shift 2
            ;;
        -e|--event)
            EVENT_ID="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -x|--execute)
            DRY_RUN=false
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validation
if [ -z "$EVENT_ID" ]; then
    echo -e "${RED}Error: Event ID is required. Use --event <event_id> or --event all${NC}"
    exit 1
fi

# Show configuration
echo -e "${BLUE}Migration Configuration:${NC}"
echo "  Index: $INDEX_NAME"
echo "  Event: $EVENT_ID"
echo "  Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN (preview only)" || echo "EXECUTE (will modify data)")"
echo "  Batch Size: $BATCH_SIZE"
echo

# Safety check for execute mode
if [ "$DRY_RUN" = false ]; then
    echo -e "${YELLOW}WARNING: This will modify metadata for vectors in Pinecone!${NC}"
    echo -e "${YELLOW}Make sure you have backups and understand the changes.${NC}"
    echo
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Migration cancelled."
        exit 0
    fi
    echo
fi

# Build Python command
PYTHON_CMD="python scripts/migrate_metadata.py --index-name $INDEX_NAME"

if [ -n "$EVENT_ID" ]; then
    PYTHON_CMD="$PYTHON_CMD --event-id $EVENT_ID"
fi

if [ "$DRY_RUN" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --dry-run"
else
    PYTHON_CMD="$PYTHON_CMD --execute"
fi

PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"

# Run migration
echo -e "${GREEN}Starting migration...${NC}"
echo "Command: $PYTHON_CMD"
echo

if eval "$PYTHON_CMD"; then
    echo
    echo -e "${GREEN}Migration completed successfully!${NC}"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}This was a dry run - no changes were made.${NC}"
        echo "To execute the migration, run with --execute flag."
    fi
    
    # Show stats file location
    STATS_FILE=$(ls -t migration_stats_${INDEX_NAME}_*.json 2>/dev/null | head -1)
    if [ -n "$STATS_FILE" ]; then
        echo "Migration stats saved to: $STATS_FILE"
    fi
else
    echo -e "${RED}Migration failed!${NC}"
    exit 1
fi