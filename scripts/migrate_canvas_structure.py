#!/usr/bin/env python3
"""
Migration script for canvas analysis folder restructure.

Migrates existing analysis documents from:
  organizations/{org}/agents/{agent}/_canvas/{event}_{mode}.md
To:
  organizations/{org}/agents/{agent}/_canvas/mlp/mlp-latest/{event}_{mode}.md

Usage:
  python scripts/migrate_canvas_structure.py --agent river --dry-run
  python scripts/migrate_canvas_structure.py --agent river
  python scripts/migrate_canvas_structure.py --all
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.s3_utils import get_s3_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "aiademomagicaudio")
S3_ORG = os.getenv("S3_ORGANIZATION", "river")


def list_existing_analysis_docs(s3_client, agent_name: str):
    """
    List all existing analysis documents in the old flat structure.

    Returns:
        List of dict with 'key', 'size', 'last_modified'
    """
    prefix = f"organizations/{S3_ORG}/agents/{agent_name}/_canvas/"

    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if 'Contents' not in response:
            logger.info(f"No existing analysis docs found for agent '{agent_name}'")
            return []

        # Filter to only .md files in the root _canvas/ folder (not in subfolders)
        docs = []
        for obj in response['Contents']:
            key = obj['Key']

            # Skip if already in new structure (contains /mlp/)
            if '/mlp/' in key:
                continue

            # Skip if in docs/ folder
            if '/_canvas/docs/' in key:
                continue

            # Only include .md files at the root _canvas/ level
            # Format: organizations/river/agents/agent_name/_canvas/0000_mirror.md
            relative_path = key.replace(prefix, '')
            if relative_path.endswith('.md') and '/' not in relative_path:
                docs.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })

        logger.info(f"Found {len(docs)} analysis docs to migrate for agent '{agent_name}'")
        return docs

    except Exception as e:
        logger.error(f"Error listing existing docs for {agent_name}: {e}", exc_info=True)
        return []


def migrate_doc(s3_client, old_key: str, dry_run: bool = False):
    """
    Migrate a single analysis document to new structure.

    Args:
        s3_client: S3 client
        old_key: Old S3 key path
        dry_run: If True, only log what would be done

    Returns:
        True if successful, False otherwise
    """
    # Parse old key to extract agent, event, and mode
    # Format: organizations/river/agents/agent_name/_canvas/0000_mirror.md
    try:
        parts = old_key.split('/')
        agent_idx = parts.index('agents') + 1
        agent_name = parts[agent_idx]

        filename = os.path.basename(old_key)  # e.g., "0000_mirror.md"
        name_parts = filename.replace('.md', '').split('_')

        # Handle both old formats:
        # - 0000_mirror.md (current)
        # - 0000_mirror_previous.md (previous)
        if len(name_parts) >= 2:
            event_id = name_parts[0]
            mode = name_parts[1]
            is_previous = 'previous' in name_parts

            if is_previous:
                # Migrate previous to mlp-previous with today's date
                date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
                new_key = f"organizations/{S3_ORG}/agents/{agent_name}/_canvas/mlp/mlp-previous/{event_id}_{mode}_{date_str}.md"
                operation = "previous → mlp-previous (with date)"
            else:
                # Migrate current to mlp-latest
                new_key = f"organizations/{S3_ORG}/agents/{agent_name}/_canvas/mlp/mlp-latest/{event_id}_{mode}.md"
                operation = "current → mlp-latest"

            if dry_run:
                logger.info(f"[DRY RUN] Would migrate: {old_key} → {new_key} ({operation})")
                return True

            # Copy to new location
            s3_client.copy_object(
                Bucket=S3_BUCKET,
                CopySource={'Bucket': S3_BUCKET, 'Key': old_key},
                Key=new_key
            )
            logger.info(f"Migrated: {filename} → {new_key} ({operation})")

            # Optional: Keep old file as backup (comment out to delete old files)
            # s3_client.delete_object(Bucket=S3_BUCKET, Key=old_key)
            # logger.info(f"Deleted old file: {old_key}")

            return True

        else:
            logger.warning(f"Could not parse filename: {filename}")
            return False

    except Exception as e:
        logger.error(f"Error migrating {old_key}: {e}", exc_info=True)
        return False


def migrate_agent(agent_name: str, dry_run: bool = False):
    """
    Migrate all analysis documents for a single agent.

    Args:
        agent_name: Agent name
        dry_run: If True, only log what would be done

    Returns:
        Number of docs successfully migrated
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Migrating canvas analysis docs for agent: {agent_name}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"{'='*60}\n")

    s3_client = get_s3_client()
    if not s3_client:
        logger.error("Failed to get S3 client")
        return 0

    # List existing docs
    docs = list_existing_analysis_docs(s3_client, agent_name)

    if not docs:
        logger.info(f"No docs to migrate for agent '{agent_name}'")
        return 0

    # Migrate each doc
    success_count = 0
    for doc in docs:
        if migrate_doc(s3_client, doc['key'], dry_run=dry_run):
            success_count += 1

    logger.info(f"\nMigration complete for '{agent_name}': {success_count}/{len(docs)} docs migrated")
    return success_count


def list_all_agents():
    """
    List all agents that have canvas analysis docs.

    Returns:
        List of agent names
    """
    s3_client = get_s3_client()
    if not s3_client:
        logger.error("Failed to get S3 client")
        return []

    prefix = f"organizations/{S3_ORG}/agents/"

    try:
        # Use paginator to handle large number of agents
        paginator = s3_client.get_paginator('list_objects_v2')
        agent_names = set()

        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix_obj in page['CommonPrefixes']:
                    # Extract agent name from prefix
                    # Format: organizations/river/agents/agent_name/
                    agent_prefix = prefix_obj['Prefix']
                    agent_name = agent_prefix.rstrip('/').split('/')[-1]

                    # Check if this agent has _canvas folder
                    canvas_prefix = f"{agent_prefix}_canvas/"
                    check_response = s3_client.list_objects_v2(
                        Bucket=S3_BUCKET,
                        Prefix=canvas_prefix,
                        MaxKeys=1
                    )
                    if 'Contents' in check_response:
                        agent_names.add(agent_name)

        agents = sorted(list(agent_names))
        logger.info(f"Found {len(agents)} agents with canvas analysis docs: {agents}")
        return agents

    except Exception as e:
        logger.error(f"Error listing agents: {e}", exc_info=True)
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Migrate canvas analysis documents to new folder structure'
    )
    parser.add_argument(
        '--agent',
        type=str,
        help='Agent name to migrate (e.g., "river")'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Migrate all agents'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - show what would be done without making changes'
    )

    args = parser.parse_args()

    if not args.agent and not args.all:
        parser.error("Must specify either --agent <name> or --all")

    if args.all:
        logger.info("Migrating all agents...")
        agents = list_all_agents()

        if not agents:
            logger.info("No agents found with canvas analysis docs")
            return

        total_migrated = 0
        for agent in agents:
            count = migrate_agent(agent, dry_run=args.dry_run)
            total_migrated += count

        logger.info(f"\n{'='*60}")
        logger.info(f"Total migration complete: {total_migrated} docs migrated across {len(agents)} agents")
        logger.info(f"{'='*60}")

    else:
        migrate_agent(args.agent, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
