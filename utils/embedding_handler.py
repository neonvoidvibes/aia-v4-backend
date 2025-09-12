"""Utilities for generating and managing document embeddings."""
import os
import sys
import logging
import re
import urllib.parse
import traceback
import yaml
from typing import List, Optional, Dict, Any

# Attempt early tiktoken import
try: import tiktoken; logging.getLogger(__name__).debug("Imported tiktoken early.")
except Exception as e: logging.getLogger(__name__).warning(f"Early tiktoken import failed: {e}")

# Langchain imports
from langchain_openai import OpenAIEmbeddings
# Use both splitter types for different document structures
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
# Document object for metadata handling with splitter
from langchain_core.documents import Document

# Local imports
from .pinecone_utils import init_pinecone
import pinecone

# Configure logging
logger = logging.getLogger(__name__)

def sanitize_for_pinecone_id(input_string: str) -> str:
    """Sanitizes a string using URL encoding for Pinecone ID compliance."""
    sanitized = urllib.parse.quote_plus(input_string.encode('utf-8'))
    max_len = 512
    if len(sanitized) > max_len:
        logger.warning(f"Sanitized ID >{max_len} chars. Truncating: {input_string}")
        sanitized = sanitized[:max_len]
    if not sanitized:
        logger.error(f"Empty sanitized ID for: {input_string}"); return "sanitized_empty"
    return sanitized

def _extract_yaml_from_enriched_log(log: str) -> Dict[str, Any]:
    """Extracts the YAML frontmatter from the enriched log and parses it."""
    try:
        match = re.search(r'---\s*\n(.*?)\n---', log, re.DOTALL)
        if match:
            frontmatter_str = match.group(1)
            # Use safe_load to parse the YAML string
            parsed_yaml = yaml.safe_load(frontmatter_str)
            if isinstance(parsed_yaml, dict):
                return parsed_yaml
            else:
                logger.warning(f"Parsed YAML is not a dictionary: {parsed_yaml}")
    except yaml.YAMLError as e:
        logger.warning(f"Could not parse YAML from enriched log frontmatter: {e}")
    except Exception as e:
        logger.warning(f"An unexpected error occurred during YAML extraction: {e}")
    return {}

class EmbeddingHandler:
    """Handles document embedding generation and storage."""

    def __init__(
        self,
        index_name: str = "river",
        namespace: Optional[str] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 150
    ):
        """Initialize embedding handler."""
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model_name = "text-embedding-ada-002"

        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
            logger.info(f"EmbeddingHandler: Initialized OpenAIEmbeddings model '{self.embedding_model_name}'.")
        except Exception as e:
            logger.error(f"EmbeddingHandler: Failed OpenAIEmbeddings init: {e}", exc_info=True)
            raise RuntimeError("Failed OpenAIEmbeddings init") from e

        # 1. Initialize Markdown splitter for "Intelligent Logs"
        headers_to_split_on = [("###", "Header 3")]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        logger.info("EmbeddingHandler: Initialized MarkdownHeaderTextSplitter.")

        # 2. Initialize Recursive splitter for general purpose docs (PDFs, plain text)
        # Use tiktoken to measure length in tokens, not characters, for better embedding performance.
        try:
            self.tiktoken_encoding = tiktoken.encoding_for_model(self.embedding_model_name)
            logger.info(f"Tiktoken encoding '{self.embedding_model_name}' loaded successfully.")
            def tiktoken_len(text: str) -> int:
                return len(self.tiktoken_encoding.encode(text))
            self.length_function = tiktoken_len
            # Adjust chunk_size to be token-based
            token_chunk_size = 1000
            token_chunk_overlap = 100
        except Exception as e:
            logger.warning(f"Tiktoken init failed: {e}. Falling back to character count for splitting.")
            self.length_function = len
            token_chunk_size = chunk_size
            token_chunk_overlap = chunk_overlap

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=token_chunk_overlap,
            length_function=self.length_function,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"EmbeddingHandler: Initialized RecursiveCharacterTextSplitter (chunk={token_chunk_size}, overlap={token_chunk_overlap}, method={'tokens' if self.length_function is not len else 'characters'}).")

        self.pc = init_pinecone()
        if not self.pc:
            logger.error("EmbeddingHandler: Failed to initialize Pinecone client.")
            self.index = None
        else:
            try:
                # Check if the index exists without creating it.
                if index_name in self.pc.list_indexes().names():
                    self.index = self.pc.Index(index_name)
                    logger.info(f"EmbeddingHandler: Successfully connected to existing Pinecone index '{index_name}'.")
                else:
                    logger.warning(f"EmbeddingHandler: Pinecone index '{index_name}' does not exist. Operations requiring an index will be skipped.")
                    self.index = None
            except Exception as e:
                logger.error(f"EmbeddingHandler: Error checking for Pinecone index '{index_name}': {e}", exc_info=True)
                self.index = None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        try:
            if not hasattr(self, 'embeddings') or self.embeddings is None: logger.error("Embeddings obj missing."); return None
            return self.embeddings.embed_query(text)
        except Exception as e: logger.error(f"Embedding generation error: {e}", exc_info=True); return None

    def embed_and_upsert(
        self,
        content: str,
        metadata: Dict[str, Any], # Base metadata passed in
        batch_size: int = 100
    ) -> bool:
        """Split, embed, and upsert document content."""
        try:
            original_file_name = metadata.get('file_name')
            if not original_file_name: logger.error("Missing 'file_name' in metadata."); return False

            # Extract metadata from the YAML frontmatter
            enriched_metadata = _extract_yaml_from_enriched_log(content)
            is_core_memory_log = enriched_metadata.get('core_memory', False)
            triplets = enriched_metadata.get('triplets', [])

            if is_core_memory_log:
                logger.info(f"Core memory flag found in '{original_file_name}'. All chunks will be marked as core.")
            if triplets:
                logger.info(f"Found {len(triplets)} triplets in '{original_file_name}'.")

            # Unified splitting logic for all documents.
            source_type = metadata.get('source', 'default')
            logger.info(f"Processing document '{original_file_name}' with token-based splitter.")
            documents = self.recursive_splitter.create_documents([content], metadatas=[metadata])

            if not documents: logger.warning(f"No documents/chunks created for: {original_file_name}"); return False
            logger.info(f"Split '{original_file_name}' into {len(documents)} documents/chunks using '{source_type}' strategy.")

            vectors_to_upsert = []
            embedding_failed_count = 0
            for i, doc in enumerate(documents):
                chunk_content = doc.page_content
                chunk_metadata = doc.metadata

                vector = self.generate_embedding(chunk_content)
                if not vector:
                    logger.warning(f"Embedding failed for chunk {i} of {original_file_name}. Skipping.")
                    embedding_failed_count += 1
                    continue

                file_name_for_id = chunk_metadata.get('file_name', original_file_name)
                page_number = chunk_metadata.get('page_number') # Get the page number
                sanitized_id_part = sanitize_for_pinecone_id(file_name_for_id)
                
                # Create a more robust, unique ID
                if page_number is not None:
                    # For PDFs, use the page number to ensure uniqueness
                    vector_id = f"{sanitized_id_part}_page-{page_number}_chunk-{i}"
                else:
                    # Fallback for non-PDF files
                    vector_id = f"{sanitized_id_part}_{i}"

                # Prepare metadata for Pinecone
                from datetime import datetime, timezone
                pinecone_metadata = {
                    **chunk_metadata,
                    'content': chunk_content,
                    'chunk_index': i,
                    'total_chunks': len(documents),
                    'supabase_log_id': metadata.get('supabase_log_id', -1),
                    'source_identifier': metadata.get('source_identifier', 'unknown'),
                    'is_core_memory': is_core_memory_log or metadata.get('is_core_memory', False), # Preserve CLI flag or use log-level flag
                    'triplets': triplets, # Add the extracted triplets
                }
                # Ensure key fields exist for retrieval policy compatibility
                if 'agent_name' not in pinecone_metadata:
                    pinecone_metadata['agent_name'] = metadata.get('agent_name', 'unknown')
                # Event awareness defaults to '0000' (shared)
                if 'event_id' not in pinecone_metadata:
                    pinecone_metadata['event_id'] = metadata.get('event_id', '0000')
                # Ensure transcript scoping metadata present
                if 'transcript' not in pinecone_metadata:
                    pinecone_metadata['transcript'] = str(metadata.get('event_id', '0000'))
                # Standardize source type
                if 'source_type' not in pinecone_metadata:
                    # Map any legacy 'source' to a standard type when sensible
                    src = metadata.get('source', 'manual_upload')
                    default_type = 'doc'
                    if 'transcript' in src:
                        default_type = 'transcript'
                    elif 'chat' in src:
                        default_type = 'chat'
                    pinecone_metadata['source_type'] = metadata.get('source_type', default_type)
                # Created at timestamp
                if 'created_at' not in pinecone_metadata:
                    pinecone_metadata['created_at'] = datetime.now(timezone.utc).isoformat()
                # Doc/Chunk identifiers
                if 'doc_id' not in pinecone_metadata:
                    pinecone_metadata['doc_id'] = metadata.get('source_identifier') or metadata.get('file_name') or 'unknown'
                pinecone_metadata['chunk_id'] = i

                vectors_to_upsert.append((vector_id, vector, pinecone_metadata))

            if embedding_failed_count > 0: logger.warning(f"{embedding_failed_count} chunks failed embedding for {original_file_name}")
            if not vectors_to_upsert: logger.warning(f"No vectors to upsert for {original_file_name}"); return False

            logger.info(f"Upserting {len(vectors_to_upsert)} vectors for '{original_file_name}' (namespace: '{self.namespace}')...")
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch, namespace=self.namespace)
                    logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors). Resp: {upsert_response}")
                except Exception as upsert_e:
                    logger.error(f"Pinecone upsert error (Batch {i//batch_size + 1}, File: '{original_file_name}'): {upsert_e}")
                    failing_ids = [item[0] for item in batch]; logger.error(f"Failing IDs (sample): {failing_ids[:5]}")
                    return False # Fail operation if batch fails

            logger.info(f"Successfully embedded/upserted {len(vectors_to_upsert)} vectors for '{original_file_name}'.")
            return True

        except Exception as e:
            logger.error(f"Error embedding/upserting file '{metadata.get('file_name', '?')}': {e}", exc_info=True)
            return False

    def delete_document(self, source_identifier: str) -> bool:
        """Delete all vectors associated with a specific source_identifier."""
        if not self.index:
            logger.error("Delete failed: Pinecone index is not available.")
            return False
        try:
            logger.info(f"Attempting to delete vectors for source_identifier='{source_identifier}' in namespace '{self.namespace}'...")
            
            # Deleting by metadata filter is the correct approach for this use case.
            delete_response = self.index.delete(
                filter={"source_identifier": {"$eq": source_identifier}},
                namespace=self.namespace
            )
            logger.info(f"Delete by filter for source_identifier='{source_identifier}' completed. Response: {delete_response}")
            return True
        except pinecone.exceptions.NotFoundException:
            # This is an expected and harmless error if the namespace doesn't exist yet (e.g., first save).
            logger.debug(f"Namespace '{self.namespace}' not found during pre-emptive delete for source '{source_identifier}'. This is normal on first save.")
            return True # The desired state (no old vectors) is achieved.
        except Exception as e:
            logger.error(f"Error deleting vectors for source_identifier='{source_identifier}': {e}", exc_info=True)
            return False
