#!/usr/bin/env python3
"""
Migration script to mark all transcripts in saved/ and archive/ folders as archived in Pinecone.
This ensures they are excluded from LLM retrieval context while remaining visible in the UI.

Usage:
    python scripts/mark_saved_transcripts_archived.py [--dry-run] [--agent AGENT_NAME]
"""

import os
import sys
import argparse
import logging
from typing import List, Set

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.s3_utils import get_s3_client, list_agent_names_from_s3
from utils.pinecone_utils import get_index
from utils.embedding_handler import sanitize_for_pinecone_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def list_saved_and_archived_transcripts(agent_name: str) -> Set[str]:
    """List all transcript filenames in saved/ and archive/ folders for an agent."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("S3 client or bucket not available")
        return set()

    saved_files = set()

    # Check all events for this agent
    try:
        # List all events
        events_prefix = f"organizations/river/agents/{agent_name}/events/"
        paginator = s3.get_paginator('list_objects_v2')

        # Get all event IDs
        result = paginator.paginate(Bucket=aws_s3_bucket, Prefix=events_prefix, Delimiter='/')
        event_ids = []
        for page in result:
            for cp in page.get('CommonPrefixes', []):
                full = cp.get('Prefix', '')
                if full.startswith(events_prefix):
                    event_id = full[len(events_prefix):].strip('/')
                    if event_id:
                        event_ids.append(event_id)

        # For each event, check saved/ and archive/ folders
        for event_id in event_ids:
            for subfolder in ['saved', 'archive']:
                folder_prefix = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/{subfolder}/"
                logger.debug(f"Checking {folder_prefix}")

                for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=folder_prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            if key.endswith('.txt') and 'transcript_' in key:
                                filename = os.path.basename(key)
                                saved_files.add(filename)
                                logger.debug(f"Found archived transcript: {filename}")

        logger.info(f"Found {len(saved_files)} archived transcripts for agent '{agent_name}'")
        return saved_files

    except Exception as e:
        logger.error(f"Error listing saved/archived transcripts for {agent_name}: {e}", exc_info=True)
        return set()


def mark_transcripts_as_archived(agent_name: str, filenames: Set[str], dry_run: bool = True) -> int:
    """Mark transcript vectors in Pinecone as archived."""
    if not filenames:
        logger.info(f"No transcripts to mark for agent '{agent_name}'")
        return 0

    index = get_index("river")
    if not index:
        logger.error("Could not connect to Pinecone index 'river'")
        return 0

    updated_count = 0

    for filename in filenames:
        try:
            # Query for vectors with this filename
            fetch_response = index.query(
                vector=[0.0] * 1536,  # Dummy vector
                filter={
                    "agent_name": agent_name,
                    "file_name": filename
                },
                top_k=10000,
                include_metadata=True,
                namespace=agent_name
            )

            vector_ids = [match.id for match in fetch_response.matches]

            if vector_ids:
                logger.info(f"  {filename}: Found {len(vector_ids)} vectors")

                if not dry_run:
                    for vid in vector_ids:
                        index.update(
                            id=vid,
                            set_metadata={"is_archived": True, "storage_location": "saved_or_archive"},
                            namespace=agent_name
                        )
                    logger.info(f"  {filename}: ✓ Marked {len(vector_ids)} vectors as archived")
                else:
                    logger.info(f"  {filename}: [DRY RUN] Would mark {len(vector_ids)} vectors as archived")

                updated_count += len(vector_ids)
            else:
                logger.debug(f"  {filename}: No vectors found in Pinecone")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)

    return updated_count


def main():
    parser = argparse.ArgumentParser(description='Mark saved/archived transcripts in Pinecone')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without updating')
    parser.add_argument('--agent', type=str, help='Process only this agent (default: all agents)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MIGRATION: Mark Saved/Archived Transcripts in Pinecone")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("Running in DRY RUN mode - no changes will be made")

    # Get list of agents
    if args.agent:
        agent_names = [args.agent]
        logger.info(f"Processing single agent: {args.agent}")
    else:
        agent_names = list_agent_names_from_s3()
        if not agent_names:
            logger.error("Could not retrieve agent list from S3")
            return 1
        logger.info(f"Processing {len(agent_names)} agents")

    total_updated = 0

    for agent_name in agent_names:
        logger.info(f"\nProcessing agent: {agent_name}")
        logger.info("-" * 80)

        # Get list of saved/archived transcripts
        archived_files = list_saved_and_archived_transcripts(agent_name)

        if not archived_files:
            logger.info(f"  No archived transcripts found")
            continue

        # Mark them in Pinecone
        count = mark_transcripts_as_archived(agent_name, archived_files, dry_run=args.dry_run)
        total_updated += count

        logger.info(f"  Agent '{agent_name}': {count} vectors processed")

    logger.info("=" * 80)
    logger.info(f"MIGRATION COMPLETE")
    logger.info(f"Total vectors updated: {total_updated}")
    if args.dry_run:
        logger.info("NOTE: This was a DRY RUN. Run without --dry-run to apply changes.")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
