#!/usr/bin/env python3
"""
Metadata Migration Script for Pinecone Vector Database

This script updates existing vectors in Pinecone with enhanced metadata fields
to support improved content categorization and retrieval.

Usage:
    python scripts/migrate_metadata.py --index-name river --event-id 0000 --dry-run
    python scripts/migrate_metadata.py --index-name river --event-id 0000 --execute
    python scripts/migrate_metadata.py --index-name river --all-events --execute
"""

import argparse
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.pinecone_utils import get_index
# API key manager not needed for metadata-only migration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetadataMigrator:
    def __init__(self, index_name: str, batch_size: int = 100):
        self.index_name = index_name
        self.batch_size = batch_size
        self.index = get_index(index_name)
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'meeting_summaries': 0,
            'chat_messages': 0,
            'foundational_docs': 0,
            'temp_documents': 0,
            'other_content': 0,
            'updated': 0,
            'errors': 0,
            'skipped': 0
        }
    
    def detect_content_type(self, metadata: Dict[str, Any], vector_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect content type and generate enhanced metadata fields.
        
        Returns:
            Tuple of (content_category, enhanced_metadata_dict)
        """
        enhanced_metadata = {}
        
        # Extract existing fields
        file_name = metadata.get('file_name', '')
        event_id = metadata.get('event_id', '0000')
        
        # 1. MEETING SUMMARIES
        if self._is_meeting_summary(file_name, metadata):
            enhanced_metadata.update({
                'content_category': 'meeting_summary',
                'analysis_type': self._detect_analysis_type(file_name, metadata),
                'temporal_relevance': 'time_sensitive',
                'meeting_date': self._extract_meeting_date(file_name),
                'summary_type': self._detect_summary_type(file_name)
            })
            self.stats['meeting_summaries'] += 1
            return 'meeting_summary', enhanced_metadata
        
        # 2. CHAT MESSAGES
        elif self._is_chat_message(metadata, vector_id):
            enhanced_metadata.update({
                'content_category': 'chat_message',
                'temporal_relevance': 'time_sensitive',
                'interaction_type': self._detect_interaction_type(metadata),
                'message_date': self._extract_message_date(metadata)
            })
            self.stats['chat_messages'] += 1
            return 'chat_message', enhanced_metadata
        
        # 3. FOUNDATIONAL DOCUMENTS
        elif self._is_foundational_doc(file_name, metadata):
            enhanced_metadata.update({
                'content_category': 'foundational_document',
                'temporal_relevance': 'persistent',
                'doc_type': self._detect_doc_type(file_name, metadata),
                'scope': self._detect_scope(event_id, metadata)
            })
            
            # Set is_core_memory=True for foundational docs (override existing False values)
            # This corrects the historical mis-classification of foundational content
            enhanced_metadata['is_core_memory'] = True
                
            self.stats['foundational_docs'] += 1
            return 'foundational_document', enhanced_metadata
        
        # 4. TEMP DOCUMENTS
        elif self._is_temp_document(file_name, metadata):
            enhanced_metadata.update({
                'content_category': 'temp_document',
                'temporal_relevance': 'session_scoped',
                'access_scope': 'session_only',
                'temp_doc_mode': True,
                'session_id': self._extract_session_id(metadata),
                'expires_at': self._calculate_expiry(metadata)
            })
            self.stats['temp_documents'] = self.stats.get('temp_documents', 0) + 1
            return 'temp_document', enhanced_metadata
        
        # 5. OTHER CONTENT
        else:
            enhanced_metadata.update({
                'content_category': 'other',
                'temporal_relevance': 'contextual'
            })
            self.stats['other_content'] += 1
            return 'other', enhanced_metadata
    
    def _is_meeting_summary(self, file_name: str, metadata: Dict[str, Any]) -> bool:
        """Detect if content is a meeting summary from our multi-agent pipeline."""
        # Only match specific patterns from our transcript pipeline
        pipeline_indicators = [
            'transcript_D', 'summary_D',  # Our date pattern: D20250812-T080755
            '_full.md', '_business_reality.md', '_context.md'  # Our layer files
        ]
        
        # Check for pipeline-generated files
        file_name_lower = file_name.lower()
        pipeline_match = any(indicator in file_name_lower for indicator in pipeline_indicators)
        
        # Also check metadata for pipeline markers
        source = metadata.get('source', '')
        source_type = metadata.get('source_type', '')
        pipeline_version = metadata.get('pipeline_version', '')
        
        metadata_match = (
            source in ['transcript_full', 'meeting_summary'] or
            source_type in ['summary', 'transcript'] or
            'business_first' in pipeline_version
        )
        
        return pipeline_match or metadata_match
    
    def _is_chat_message(self, metadata: Dict[str, Any], vector_id: str) -> bool:
        """Detect if content is a chat message."""
        # Look for chat-specific metadata fields
        chat_indicators = [
            'message_id', 'user_id', 'timestamp', 'chat_id',
            'conversation_id', 'thread_id'
        ]
        
        return any(field in metadata for field in chat_indicators) or 'chat' in vector_id.lower()
    
    def _is_foundational_doc(self, file_name: str, metadata: Dict[str, Any]) -> bool:
        """Detect if content is a foundational document."""
        # File-based indicators
        foundational_indicators = [
            'systemprompt', 'context', 'instructions', 'guidelines',
            'framework', 'template', 'config', 'strategy', 'definitions',
            'whitepaper', 'white_paper', 'manual', 'handbook', 'reference'
        ]
        
        file_name_lower = file_name.lower()
        filename_match = any(indicator in file_name_lower for indicator in foundational_indicators)
        
        # Metadata-based indicators for foundational content
        source = metadata.get('source', '')
        source_type = metadata.get('source_type', '')
        is_core_memory = metadata.get('is_core_memory', False)
        
        # Core logic: Manual uploads are foundational content (permanent knowledge for the agent)
        # EXCEPT for chat_memory files which are conversational, not foundational documents
        is_manual_upload = source == 'manual_upload'
        is_chat_file = 'chat_memory' in file_name_lower
        
        # Foundational content patterns:
        # 1. Explicitly marked as core memory
        # 2. Documents with foundational source types  
        # 3. Manual uploads that aren't chat files (these are curated permanent knowledge)
        # 4. Files with foundational keywords in filename
        metadata_match = (
            is_core_memory or
            source_type in ['doc', 'reference', 'manual'] or
            source in ['knowledge_base', 'reference_material'] or
            (is_manual_upload and not is_chat_file)
        )
        
        return filename_match or metadata_match
    
    def _is_temp_document(self, file_name: str, metadata: Dict[str, Any]) -> bool:
        """Detect if content is a temporary document for chat sessions."""
        # Check metadata flags for temp documents
        is_core_memory = metadata.get('is_core_memory', True)  # Default True for existing content
        temp_doc_mode = metadata.get('temp_doc_mode', False)
        access_scope = metadata.get('access_scope', '')
        
        # Explicit temp document markers
        explicit_temp = (
            not is_core_memory and temp_doc_mode or
            access_scope == 'session_only'
        )
        
        # File pattern indicators for temporary documents
        temp_patterns = [
            'temp_', 'temporary_', 'session_', 'chat_upload_',
            '_temp.', '_temporary.', '_session.'
        ]
        
        file_name_lower = file_name.lower()
        pattern_match = any(pattern in file_name_lower for pattern in temp_patterns)
        
        return explicit_temp or pattern_match
    
    def _extract_session_id(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract session ID from metadata for temp documents."""
        # Check for existing session identifiers
        session_indicators = [
            'session_id', 'chat_session_id', 'conversation_id', 
            'thread_id', 'chat_id'
        ]
        
        for indicator in session_indicators:
            if indicator in metadata:
                return str(metadata[indicator])
        
        # Try to extract from event_id if it looks like a session ID
        event_id = metadata.get('event_id')
        if event_id and (event_id.startswith('chat_') or event_id.startswith('session_')):
            return event_id
            
        return None
    
    def _calculate_expiry(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Calculate expiry timestamp for temp documents."""
        # Check if expiry is already set
        if 'expires_at' in metadata:
            return metadata['expires_at']
        
        # For now, don't auto-calculate expiry from migration
        # This will be handled when creating new temp docs
        return None
    
    def _detect_analysis_type(self, file_name: str, metadata: Dict[str, Any]) -> str:
        """Detect the type of analysis (multi_agent, single_layer, etc.)."""
        if '_full.md' in file_name or 'Layer' in str(metadata.get('content', '')):
            return 'multi_agent'
        elif '_business_reality.md' in file_name:
            return 'single_layer'
        else:
            return 'summary'
    
    def _detect_summary_type(self, file_name: str) -> str:
        """Detect specific summary type."""
        if '_full.md' in file_name:
            return 'comprehensive'
        elif '_business_reality.md' in file_name:
            return 'business_focused'
        elif '_context.md' in file_name:
            return 'contextual'
        else:
            return 'standard'
    
    def _detect_interaction_type(self, metadata: Dict[str, Any]) -> str:
        """Detect type of chat interaction."""
        # Could be enhanced based on actual chat metadata structure
        if 'system' in str(metadata.get('role', '')).lower():
            return 'system_message'
        elif 'user' in str(metadata.get('role', '')).lower():
            return 'user_message'
        elif 'assistant' in str(metadata.get('role', '')).lower():
            return 'assistant_message'
        else:
            return 'chat_message'
    
    def _detect_doc_type(self, file_name: str, metadata: Dict[str, Any]) -> str:
        """Detect foundational document type."""
        if 'systemprompt' in file_name.lower():
            return 'system_prompt'
        elif 'context' in file_name.lower():
            return 'context_document'
        elif 'instructions' in file_name.lower():
            return 'instructions'
        elif 'framework' in file_name.lower():
            return 'framework'
        else:
            return 'reference'
    
    def _detect_scope(self, event_id: str, metadata: Dict[str, Any]) -> str:
        """Detect document scope (global, event_specific, etc.)."""
        if event_id == '0000' or event_id == 'global':
            return 'global'
        else:
            return 'event_specific'
    
    def _extract_meeting_date(self, file_name: str) -> Optional[str]:
        """Extract meeting date from filename pattern."""
        # Pattern: D20250812-T080755
        date_match = re.search(r'_D(\d{8})-T(\d{6})_', file_name)
        if date_match:
            date_str, time_str = date_match.groups()
            try:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                dt = datetime(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        return None
    
    def _extract_message_date(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract message date from metadata."""
        timestamp = metadata.get('timestamp') or metadata.get('created_at')
        if timestamp:
            try:
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                else:
                    dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        
        return None
    
    def get_vectors_to_migrate(self, event_id: Optional[str] = None) -> List[Tuple[str, str]]:
        """Get list of (vector_id, namespace) tuples that need migration."""
        vector_tuples = []
        
        try:
            # Get all namespaces from index stats
            stats = self.index.describe_index_stats()
            namespaces = stats['namespaces']
            
            logger.info(f"Found {len(namespaces)} namespaces to scan: {list(namespaces.keys())}")
            
            for namespace_name, ns_stats in namespaces.items():
                vector_count = ns_stats['vector_count']
                if vector_count == 0:
                    continue
                    
                logger.info(f"Scanning namespace '{namespace_name}' with {vector_count} vectors...")
                
                try:
                    # Query vectors in this namespace
                    filter_dict = {}
                    if event_id and event_id != 'all':
                        filter_dict['event_id'] = event_id
                    
                    # Use query with dummy vector to get all vectors in namespace
                    query_response = self.index.query(
                        vector=[0.0] * 1536,  # Dummy query vector
                        filter=filter_dict,
                        namespace=namespace_name,
                        top_k=min(10000, vector_count * 2),  # Get all vectors with buffer
                        include_metadata=True
                    )
                    
                    namespace_vectors = []
                    for match in query_response.matches:
                        namespace_vectors.append((match.id, namespace_name))
                    
                    vector_tuples.extend(namespace_vectors)
                    logger.info(f"Found {len(namespace_vectors)} vectors in namespace '{namespace_name}'")
                    
                except Exception as e:
                    logger.error(f"Error querying namespace '{namespace_name}': {e}")
                    continue
            
            logger.info(f"Total vectors found across all namespaces: {len(vector_tuples)}")
            
        except Exception as e:
            logger.error(f"Error getting namespaces: {e}")
            
        return vector_tuples
    
    def migrate_batch(self, vector_tuples: List[Tuple[str, str]], dry_run: bool = True) -> int:
        """Migrate a batch of vectors from potentially different namespaces."""
        updated_count = 0
        
        # Group vectors by namespace for efficient fetching
        namespace_groups = {}
        for vector_id, namespace in vector_tuples:
            if namespace not in namespace_groups:
                namespace_groups[namespace] = []
            namespace_groups[namespace].append(vector_id)
        
        for namespace, vector_ids in namespace_groups.items():
            try:
                # Fetch current vectors from this namespace
                fetch_response = self.index.fetch(ids=vector_ids, namespace=namespace)
                
                updates = []
                
                for vector_id, vector_data in fetch_response.vectors.items():
                    try:
                        current_metadata = vector_data.metadata or {}
                        
                        # Check if already has new metadata fields
                        if 'content_category' in current_metadata:
                            logger.debug(f"Vector {vector_id} (ns: {namespace}) already migrated, skipping")
                            self.stats['skipped'] += 1
                            continue
                        
                        # Detect content type and generate enhanced metadata
                        content_category, enhanced_metadata = self.detect_content_type(
                            current_metadata, vector_id
                        )
                        
                        # Merge with existing metadata
                        new_metadata = {**current_metadata, **enhanced_metadata}
                        
                        if not dry_run:
                            updates.append({
                                'id': vector_id,
                                'metadata': new_metadata
                            })
                        
                        updated_count += 1
                        self.stats['updated'] += 1
                        
                        logger.debug(f"Vector {vector_id} (ns: {namespace}): {content_category} -> {enhanced_metadata}")
                        
                    except Exception as e:
                        logger.error(f"Error processing vector {vector_id} in namespace {namespace}: {e}")
                        self.stats['errors'] += 1
                
                # Perform batch update for this namespace
                if updates and not dry_run:
                    self.index.update(updates=updates, namespace=namespace)
                    logger.info(f"Updated {len(updates)} vectors in namespace '{namespace}'")
                
            except Exception as e:
                logger.error(f"Error in batch migration for namespace '{namespace}': {e}")
                self.stats['errors'] += len(vector_ids)
        
        return updated_count
    
    def migrate_all(self, event_id: Optional[str] = None, dry_run: bool = True) -> Dict[str, int]:
        """Migrate all vectors matching criteria."""
        logger.info(f"Starting migration for index '{self.index_name}'")
        logger.info(f"Event ID filter: {event_id or 'all events'}")
        logger.info(f"Dry run: {dry_run}")
        
        # Get all vector tuples to migrate
        vector_tuples = self.get_vectors_to_migrate(event_id)
        
        if not vector_tuples:
            logger.warning("No vectors found to migrate")
            return self.stats
        
        # Process in batches
        total_batches = (len(vector_tuples) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(vector_tuples), self.batch_size):
            batch = vector_tuples[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} vectors)")
            
            self.migrate_batch(batch, dry_run)
            self.stats['total_processed'] += len(batch)
        
        # Print summary
        logger.info("Migration completed!")
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Meeting summaries: {self.stats['meeting_summaries']}")
        logger.info(f"Chat messages: {self.stats['chat_messages']}")
        logger.info(f"Foundational docs: {self.stats['foundational_docs']}")
        logger.info(f"Temp documents: {self.stats['temp_documents']}")
        logger.info(f"Other content: {self.stats['other_content']}")
        logger.info(f"Updated: {self.stats['updated']}")
        logger.info(f"Skipped (already migrated): {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(description='Migrate Pinecone metadata to enhanced schema')
    parser.add_argument('--index-name', required=True, help='Pinecone index name')
    parser.add_argument('--event-id', help='Specific event ID to migrate (or "all" for all events)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without executing')
    parser.add_argument('--execute', action='store_true', help='Execute the migration')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate arguments
    if not args.dry_run and not args.execute:
        logger.error("Must specify either --dry-run or --execute")
        return 1
    
    if args.dry_run and args.execute:
        logger.error("Cannot specify both --dry-run and --execute")
        return 1
    
    try:
        # Create migrator
        migrator = MetadataMigrator(args.index_name, args.batch_size)
        
        # Run migration
        stats = migrator.migrate_all(
            event_id=args.event_id,
            dry_run=args.dry_run
        )
        
        # Save stats to file
        stats_file = f"migration_stats_{args.index_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Migration stats saved to: {stats_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())