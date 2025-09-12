#!/bin/bash

# Migration runner script for text-embedding-3-small migration
# This script follows the phased approach outlined in the migration plan

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATION_SCRIPT="$SCRIPT_DIR/migrate_to_text_embedding_3_small.py"

echo "==============================================="
echo "Text-Embedding-3-Small Migration Runner"
echo "==============================================="

# Check if migration script exists
if [[ ! -f "$MIGRATION_SCRIPT" ]]; then
    echo "ERROR: Migration script not found at $MIGRATION_SCRIPT"
    exit 1
fi

# Function to run migration command
run_migration() {
    local cmd="$1"
    local description="$2"
    
    echo ""
    echo ">>> $description"
    echo ">>> Command: python $cmd"
    echo ""
    
    if python "$MIGRATION_SCRIPT" $cmd; then
        echo "✅ SUCCESS: $description"
    else
        echo "❌ FAILED: $description"
        exit 1
    fi
}

# Phase 1: Test namespace migration
echo "Phase 1: Testing with _test namespace"
echo "======================================"

# Dry run first
run_migration "--test-only --dry-run" "Dry run for _test namespace"

# Confirm before proceeding
echo ""
read -p "Proceed with _test namespace migration? (y/N): " confirm
if [[ $confirm =~ ^[Yy]$ ]]; then
    run_migration "--test-only" "Migrate _test namespace"
    
    # Verify test migration
    echo ""
    echo "Verifying _test namespace migration..."
    test_namespaces=$(python "$MIGRATION_SCRIPT" --dry-run --test-only 2>/dev/null | grep "Namespaces to migrate" | cut -d':' -f2 | tr -d ' []')
    
    if [[ -n "$test_namespaces" ]]; then
        # Get first test namespace for verification
        first_test_ns=$(echo "$test_namespaces" | cut -d',' -f1 | tr -d "'")
        run_migration "--namespace $first_test_ns --verify-only" "Verify _test namespace migration"
    fi
    
    echo ""
    echo "✅ Phase 1 completed successfully!"
    echo ""
    
    # Phase 2: All namespaces
    echo "Phase 2: All namespaces migration"
    echo "=================================="
    
    read -p "Proceed with full migration of all namespaces? (y/N): " confirm_all
    if [[ $confirm_all =~ ^[Yy]$ ]]; then
        # Dry run for all
        run_migration "--dry-run" "Dry run for all namespaces"
        
        echo ""
        read -p "Final confirmation - migrate ALL namespaces? (y/N): " final_confirm
        if [[ $final_confirm =~ ^[Yy]$ ]]; then
            run_migration "" "Migrate all namespaces"
            echo ""
            echo "🎉 Migration completed successfully!"
            echo ""
            echo "Next steps:"
            echo "1. Update Supabase mappings to reflect the new embedding model"
            echo "2. Test RAG queries to verify improved recall"
            echo "3. Monitor application performance"
        else
            echo "Migration cancelled by user"
            exit 1
        fi
    else
        echo "Phase 2 skipped by user"
        exit 0
    fi
else
    echo "Migration cancelled by user"
    exit 1
fi