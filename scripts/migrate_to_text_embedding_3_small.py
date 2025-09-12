#!/usr/bin/env python3
"""
Migration script to migrate from text-embedding-ada-002 to text-embedding-3-small in place.
Phase 1: Migrate _test namespace first
Phase 2: Migrate all namespaces after validation
"""
import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
import itertools
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressSpinner:
    """ASCII spinner for progress indication"""
    def __init__(self, message="Processing"):
        self.message = message
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.busy = False
        self.delay = 0.1
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.stop()
        
    def start(self):
        self.busy = True
        threading.Thread(target=self._spin).start()
        
    def stop(self):
        self.busy = False
        print('\r', end='')  # Clear the line
        
    def _spin(self):
        while self.busy:
            print(f'\r{next(self.spinner)} {self.message}...', end='', flush=True)
            time.sleep(self.delay)

# Load environment variables
load_dotenv()

# Configuration  
INDEX_NAME = "river"  # Hardcoded to match API server
BATCH_SIZE = 100
NEW_MODEL = "text-embedding-3-small"
VECTOR_SPACE_TAG = "te3s-2025-09"
EMBED_VERSION = "v1"

# Initialize clients
pc = Pinecone()
client = OpenAI()

def batched(seq, n):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    response = client.embeddings.create(model=NEW_MODEL, input=texts)
    return [d.embedding for d in response.data]

class SimpleResponse:
    def __init__(self, ids_list, cursor=None):
        self.ids = ids_list
        self.cursor = cursor

def list_ids(index, namespace: str, cursor: Optional[str] = None, limit: int = 50):
    """List vector IDs in a namespace - ultra simplified version."""
    try:
        # Use query with no filter to get basic stats first
        stats = index.describe_index_stats()
        ns_info = stats.get('namespaces', {}).get(namespace)
        if not ns_info:
            logger.warning(f"Namespace {namespace} not found in stats")
            return SimpleResponse([], None)
        
        vector_count = ns_info.get('vector_count', 0)
        if vector_count == 0:
            logger.info(f"Namespace {namespace} is empty")
            return SimpleResponse([], None)
        
        # For small namespaces, get all IDs at once using query
        # This is a workaround for list() API issues
        # For larger namespaces, we need to get all vectors, not just limit
        max_query_limit = 10000  # Pinecone's max top_k limit
        actual_limit = min(max_query_limit, vector_count)
        
        query_response = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=actual_limit,
            namespace=namespace,
            include_metadata=False,
            include_values=False
        )
        
        ids = [match.id for match in query_response.matches]
        return SimpleResponse(ids, None)
        
    except Exception as e:
        logger.error(f"Failed to list IDs in namespace {namespace}: {e}")
        return SimpleResponse([], None)

@retry(wait=wait_exponential_jitter(0.5, 10), stop=stop_after_attempt(7))
def fetch(index, namespace: str, ids: List[str]):
    """Fetch vectors by IDs with retry logic."""
    return index.fetch(ids=ids, namespace=namespace)

@retry(wait=wait_exponential_jitter(0.5, 10), stop=stop_after_attempt(7))
def upsert(index, namespace: str, vectors: List[Dict[str, Any]]):
    """Upsert vectors with retry logic."""
    return index.upsert(vectors=vectors, namespace=namespace)

def migrate_namespace(index, namespace: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Migrate a single namespace to text-embedding-3-small in place.
    
    Args:
        index: Pinecone index object
        namespace: Namespace to migrate
        dry_run: If True, only count vectors without making changes
        
    Returns:
        Dictionary with migration statistics
    """
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Starting migration for namespace: {namespace}")
    
    stats = {
        'namespace': namespace,
        'total_processed': 0,
        'total_updated': 0,
        'errors': 0,
        'already_migrated': 0,
        'missing_content': 0,
        'start_time': time.time()
    }
    
    cursor = None
    
    while True:
        try:
            # List IDs in batches
            with ProgressSpinner(f"Listing vector IDs in namespace {namespace}"):
                page = list_ids(index, namespace, cursor=cursor, limit=50)
                ids = page.ids
            
            if not ids:
                break
                
            logger.info(f"✓ Found batch of {len(ids)} vectors")
            
            # Fetch vectors
            with ProgressSpinner(f"Fetching vector metadata"):
                records = fetch(index, namespace, ids)
            
            # Process each vector
            batch_to_upsert = []
            
            # Handle FetchResponse object correctly
            if hasattr(records, 'vectors') and records.vectors:
                vectors_dict = records.vectors
            elif hasattr(records, 'get'):
                vectors_dict = records.get('vectors', {})
            else:
                vectors_dict = records if isinstance(records, dict) else {}
            
            logger.info(f"📋 Processing {len(vectors_dict)} vectors...")
            
            for i, (vid, vector_data) in enumerate(vectors_dict.items()):
                stats['total_processed'] += 1
                
                if i % 10 == 0 and i > 0:
                    print(f"  📄 Processed {i}/{len(vectors_dict)}", end='\r')
                
                # Handle different vector_data formats
                if hasattr(vector_data, 'metadata'):
                    metadata = vector_data.metadata or {}
                elif isinstance(vector_data, dict):
                    metadata = vector_data.get('metadata', {})
                else:
                    metadata = {}
                
                # Check if already migrated
                current_model = metadata.get('embed_model') if metadata else None
                if current_model == NEW_MODEL:
                    stats['already_migrated'] += 1
                    continue
                
                # Get content for re-embedding
                content = metadata.get('content') or metadata.get('text') or '' if metadata else ''
                if not content:
                    stats['missing_content'] += 1
                    logger.warning(f"⚠️  Vector {vid} has no content field, skipping")
                    continue
                
                if dry_run:
                    stats['total_updated'] += 1
                    continue
                
                # Update metadata
                updated_metadata = {
                    **metadata,
                    'embed_model': NEW_MODEL,
                    'embed_version': EMBED_VERSION,
                    'vector_space': VECTOR_SPACE_TAG
                }
                
                batch_to_upsert.append({
                    'id': vid,
                    'content': content,
                    'metadata': updated_metadata
                })
                
                # Process in batches
                if len(batch_to_upsert) >= BATCH_SIZE:
                    if not dry_run:
                        process_batch(index, namespace, batch_to_upsert, stats)
                    batch_to_upsert = []
            
            print()  # New line after progress
            
            # Process remaining batch
            if batch_to_upsert and not dry_run:
                process_batch(index, namespace, batch_to_upsert, stats)
            elif batch_to_upsert:  # dry_run
                stats['total_updated'] += len(batch_to_upsert)
                
            cursor = page.cursor
            if not cursor:
                break
                
        except Exception as e:
            logger.error(f"Error processing batch in namespace {namespace}: {e}")
            stats['errors'] += 1
    
    # Calculate final stats
    stats['end_time'] = time.time()
    stats['duration'] = stats['end_time'] - stats['start_time']
    
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Migration completed for namespace {namespace}")
    logger.info(f"Stats: {stats}")
    
    return stats

def process_batch(index, namespace: str, batch: List[Dict], stats: Dict):
    """Process a batch of vectors for re-embedding and upsert."""
    try:
        # Extract texts for embedding
        texts = [item['content'] for item in batch]
        
        # Generate new embeddings
        with ProgressSpinner(f"🧠 Generating embeddings for {len(texts)} texts"):
            embeddings = embed(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (item, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append({
                'id': item['id'],
                'values': embedding,
                'metadata': item['metadata']
            })
        
        # Upsert vectors (this overwrites existing vectors with same IDs)
        with ProgressSpinner(f"💾 Upserting {len(vectors)} vectors"):
            response = upsert(index, namespace, vectors)
        
        stats['total_updated'] += len(vectors)
        logger.info(f"✅ Successfully upserted batch of {len(vectors)} vectors")
        
    except Exception as e:
        logger.error(f"❌ Error processing batch: {e}")
        stats['errors'] += len(batch)
        raise

def verify_migration(index, namespace: str, sample_size: int = 10) -> bool:
    """Verify that migration was successful by checking a sample of vectors."""
    logger.info(f"Verifying migration for namespace {namespace}...")
    
    try:
        # Get a sample of IDs
        page = list_ids(index, namespace, limit=sample_size)
        if not page.ids:
            logger.warning(f"No vectors found in namespace {namespace}")
            return True
            
        # Fetch sample vectors
        records = fetch(index, namespace, page.ids)
        
        # Handle FetchResponse object correctly
        if hasattr(records, 'vectors') and records.vectors:
            vectors_dict = records.vectors
        elif hasattr(records, 'get'):
            vectors_dict = records.get('vectors', {})
        else:
            vectors_dict = records if isinstance(records, dict) else {}
        
        migrated_count = 0
        total_count = len(vectors_dict)
        
        for vid, vector_data in vectors_dict.items():
            # Handle different vector_data formats
            if hasattr(vector_data, 'metadata'):
                metadata = vector_data.metadata or {}
            elif isinstance(vector_data, dict):
                metadata = vector_data.get('metadata', {})
            else:
                metadata = {}
                
            if metadata.get('embed_model') == NEW_MODEL:
                migrated_count += 1
        
        success_rate = migrated_count / total_count if total_count > 0 else 0
        logger.info(f"Verification: {migrated_count}/{total_count} vectors migrated ({success_rate:.2%})")
        
        return success_rate >= 0.99  # 99% success rate required
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Pinecone vectors to text-embedding-3-small')
    parser.add_argument('--namespace', help='Specific namespace to migrate (optional)')
    parser.add_argument('--dry-run', action='store_true', help='Count vectors without making changes')
    parser.add_argument('--test-only', action='store_true', help='Migrate only _test namespace')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing migration')
    args = parser.parse_args()
    
    try:
        # Initialize index
        logger.info(f"Connecting to index: {INDEX_NAME}")
        index = pc.Index(INDEX_NAME)
        
        if args.verify_only:
            # Verification only
            if args.namespace:
                success = verify_migration(index, args.namespace)
                return 0 if success else 1
            else:
                logger.error("Please specify --namespace for verification")
                return 1
        
        # Get list of namespaces
        namespaces_response = index.list_namespaces()
        
        # Handle different response formats
        if hasattr(namespaces_response, 'namespaces'):
            all_namespaces = list(namespaces_response.namespaces)
        else:
            all_namespaces = list(namespaces_response)
        
        # Extract namespace names from response (handle both string and dict formats)
        namespace_names = []
        for ns in all_namespaces:
            if hasattr(ns, 'name'):
                # NamespaceDescription object
                namespace_names.append(ns.name)
            elif isinstance(ns, dict):
                namespace_names.append(ns.get('name', str(ns)))
            else:
                namespace_names.append(str(ns))
        
        logger.info(f"Available namespaces: {namespace_names}")
        
        if args.namespace:
            # Migrate specific namespace
            namespaces_to_migrate = [args.namespace]
        elif args.test_only:
            # Migrate only test namespaces
            namespaces_to_migrate = [ns for ns in namespace_names if '_test' in ns.lower()]
            if not namespaces_to_migrate:
                logger.error("No _test namespace found")
                return 1
        else:
            # Migrate all namespaces
            namespaces_to_migrate = namespace_names
        
        logger.info(f"Namespaces to migrate: {namespaces_to_migrate}")
        
        # Migration results
        results = []
        
        for namespace in namespaces_to_migrate:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing namespace: {namespace}")
            logger.info(f"{'='*60}")
            
            # Migrate namespace
            stats = migrate_namespace(index, namespace, dry_run=args.dry_run)
            results.append(stats)
            
            # Verify migration if not dry run
            if not args.dry_run:
                # Add delay for Pinecone consistency (metadata updates can take 10+ seconds)
                logger.info("Waiting for Pinecone consistency before verification...")
                time.sleep(15)
                verification_passed = verify_migration(index, namespace)
                stats['verification_passed'] = verification_passed
                
                if not verification_passed:
                    logger.error(f"Verification failed for namespace {namespace}")
                    return 1
        
        # Print final summary
        logger.info(f"\n{'='*60}")
        logger.info("MIGRATION SUMMARY")
        logger.info(f"{'='*60}")
        
        total_processed = sum(r['total_processed'] for r in results)
        total_updated = sum(r['total_updated'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        
        logger.info(f"Total namespaces: {len(results)}")
        logger.info(f"Total vectors processed: {total_processed}")
        logger.info(f"Total vectors updated: {total_updated}")
        logger.info(f"Total errors: {total_errors}")
        
        for result in results:
            logger.info(f"  {result['namespace']}: {result['total_updated']} updated, {result['errors']} errors")
        
        return 0 if total_errors == 0 else 1
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)