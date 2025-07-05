"""Utilities for generating and managing document embeddings."""
import os
import sys
import logging
import re
import urllib.parse
import traceback
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
from .prompts import CORE_MEMORY_CLASSIFIER_PROMPT
import pinecone
from anthropic import Anthropic

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

class EmbeddingHandler:
    """Handles document embedding generation and storage."""

    def __init__(
        self,
        index_name: str = "magicchat",
        namespace: Optional[str] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        anthropic_client: Optional[Anthropic] = None
    ):
        """Initialize embedding handler."""
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model_name = "text-embedding-ada-002"
        self.anthropic_client = anthropic_client

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

    def _is_core_memory(self, text: str) -> bool:
        """Uses an LLM to classify if a text snippet is a core memory."""
        if not self.anthropic_client:
            logger.warning("Anthropic client not available. Cannot classify core memory. Defaulting to False.")
            return False
        
        # Limit text length to avoid excessive token usage for classification
        max_chars_for_classification = 1000
        truncated_text = text[:max_chars_for_classification]

        try:
            prompt = CORE_MEMORY_CLASSIFIER_PROMPT.format(user_text=truncated_text)
            
            message = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307", # Using Haiku for speed and cost-effectiveness
                max_tokens=10, # Response is just 'true' or 'false'
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # We want a deterministic classification
            )
            
            response_text = message.content[0].text.strip().lower()
            is_core = response_text == 'true'
            logger.debug(f"Core memory classification for text snippet: '{truncated_text[:100]}...' -> Result: {is_core}")
            return is_core

        except Exception as e:
            logger.error(f"Error during core memory classification: {e}", exc_info=True)
            return False # Default to not being a core memory on error

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

            # Unified splitting logic for all documents.
            # The splitter is now token-aware, so it will handle chat logs and other docs appropriately.
            source_type = metadata.get('source', 'default')
            logger.info(f"Processing document '{original_file_name}' with token-based splitter.")
            documents = self.recursive_splitter.create_documents([content], metadatas=[metadata])

            if not documents: logger.warning(f"No documents/chunks created for: {original_file_name}"); return False
            logger.info(f"Split '{original_file_name}' into {len(documents)} documents/chunks using '{source_type}' strategy.")

            vectors_to_upsert = []
            embedding_failed_count = 0
            for i, doc in enumerate(documents):
                # Content is in doc.page_content
                # Metadata is in doc.metadata (includes base metadata + potentially splitter additions like start_index)
                chunk_content = doc.page_content
                chunk_metadata = doc.metadata

                vector = self.generate_embedding(chunk_content)
                if not vector:
                    logger.warning(f"Embedding failed for chunk {i} of {original_file_name}. Skipping.")
                    embedding_failed_count += 1
                    continue

                # Ensure file_name from metadata is used for ID generation
                file_name_for_id = chunk_metadata.get('file_name', original_file_name) # Fallback just in case
                sanitized_id_part = sanitize_for_pinecone_id(file_name_for_id)
                # Use simple index 'i' for uniqueness as splitter might not add chunk_index
                vector_id = f"{sanitized_id_part}_{i}"

                # Classify if the chunk is a core memory
                is_core = self._is_core_memory(chunk_content)

                # Prepare metadata for Pinecone - Ensure required fields are present
                # The splitter copies the base metadata, we add the content itself
                pinecone_metadata = {
                    **chunk_metadata, # Copy all metadata from the split document
                    'content': chunk_content, # Explicitly add content field
                    'chunk_index': i, # Explicitly add chunk index
                    'total_chunks': len(documents), # Add total chunks for context
                    'supabase_log_id': metadata.get('supabase_log_id', -1), # Link to agent_memory_logs table
                    'source_identifier': metadata.get('source_identifier', 'unknown'), # Link to source session
                    'is_core_memory': is_core, # Add the classification result
                }
                # Ensure agent_name is present if not added by splitter metadata copy
                if 'agent_name' not in pinecone_metadata:
                     pinecone_metadata['agent_name'] = metadata.get('agent_name', 'unknown')
                if 'source' not in pinecone_metadata:
                     pinecone_metadata['source'] = metadata.get('source', 'manual_upload') # Default source


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
