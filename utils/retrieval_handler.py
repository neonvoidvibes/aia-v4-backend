"""Utilities for retrieving relevant document context for chat."""
import logging
import traceback
import time
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

# Attempt early tiktoken import
try: import tiktoken; logging.getLogger(__name__).debug("Imported tiktoken early.")
except Exception as e: logging.getLogger(__name__).warning(f"Early tiktoken import failed: {e}")

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
from utils.pinecone_utils import init_pinecone
from anthropic import Anthropic # Import Anthropic client

logger = logging.getLogger(__name__)

# Define a default transform prompt
DEFAULT_QUERY_TRANSFORM_PROMPT = """Rewrite the following user query to be more effective for searching a vector database containing document chunks. Focus on extracting key entities (people, projects, organizations), topics, dates, and the core question intent.

**Crucially:**
1.  **Do NOT invent or add specific topics or subjects** (like 'machine learning', 'artificial intelligence', 'data science', etc.) if they are not explicitly mentioned or clearly implied in the original User Query.
2.  For very broad or vague queries like "list terms", "what are the terms", or just "terms", prioritize searching for documents explicitly related to glossaries, definitions, or "Terms and Conditions". If unsure, it's better to use the original query terms directly or slightly rephrase for clarity (e.g., "definitions of terms", "glossary of terms") rather than adding unrelated topics.
3.  Retain the core keywords and intent of the original query.

Output only the rewritten query, no preamble.

User Query: '{user_query}'

Rewritten Query:"""


class RetrievalHandler:
    """Handles document retrieval using direct Pinecone queries."""

    def __init__(
        self,
        index_name: str = "magicchat",
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        top_k: int = 10, # Keep moderate top_k for now
        anthropic_client: Optional[Anthropic] = None, # Expect client instance
        openai_api_key: Optional[str] = None # Agent-specific key
    ):
        """Initialize retrieval handler."""
        if not agent_name: raise ValueError("agent_name required")
        if not anthropic_client: raise ValueError("anthropic_client required for query transformation")

        self.index_name = index_name
        self.namespace = agent_name
        self.session_id = session_id
        self.event_id = event_id if event_id and event_id != '0000' else None
        self.top_k = top_k
        self.embedding_model_name = "text-embedding-ada-002"
        self.anthropic_client = anthropic_client # Store client instance

        try:
            # Initialize embeddings with the provided key. If None, it will fall back to the env var.
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model_name,
                api_key=openai_api_key
            )
            logger.info(f"Retriever: Initialized Embeddings model '{self.embedding_model_name}'.")
        except Exception as e: raise RuntimeError("Failed Embeddings init") from e

        pc = init_pinecone()
        if not pc:
            raise RuntimeError("Failed Pinecone init")
        
        try:
            # Check if index exists before creating an Index object
            if self.index_name not in pc.list_indexes().names():
                logger.warning(f"Retriever: Index '{self.index_name}' does not exist. RAG will be disabled for this handler.")
                self.index = None
            else:
                self.index = pc.Index(self.index_name)
                logger.info(f"Retriever: Connected to index '{self.index_name}'. Default top_k={self.top_k}")
        except Exception as e:
            logger.error(f"Retriever: Error connecting to or checking index '{self.index_name}': {e}", exc_info=True)
            self.index = None # Ensure index is None on any error

    def _transform_query(self, query: str) -> str:
        """Uses LLM to rewrite the query for better vector search."""
        logger.debug(f"Transforming query: '{query}'")
        try:
            # Use a smaller/faster model for transformation if available and cost-effective
            # model_for_transform = "claude-3-haiku-20240307"
            # For now, use the main client's default or passed model if needed
            # Note: This adds an extra LLM call.

            prompt = DEFAULT_QUERY_TRANSFORM_PROMPT.format(user_query=query)

            # Using non-streaming call for simplicity here
            message = self.anthropic_client.messages.create(
                 # Consider using a faster/cheaper model if possible for this task
                 model="claude-3-haiku-20240307",
                 max_tokens=100, # Should be short
                 messages=[{"role": "user", "content": prompt}]
            )
            transformed_query = message.content[0].text.strip()
            logger.debug(f"Transformed query: '{transformed_query}'")
            # Basic validation: if empty or just punctuation, return original
            if not transformed_query or transformed_query in ['.', '?', '!']:
                 logger.warning("Query transformation resulted in empty/trivial output. Using original.")
                 return query
            return transformed_query
        except Exception as e:
            logger.error(f"Error transforming query: {e}. Using original query.")
            return query # Fallback to original query on error

    def get_relevant_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        is_transcript: bool = False
    ) -> List[Document]:
        """Retrieve relevant document chunks, applying query transformation."""
        if not self.index:
            logger.info("Retriever: Skipping context retrieval as Pinecone index is not available.")
            return []
            
        k = top_k or self.top_k
        logger.debug(f"Retriever: Original query: '{query[:100]}...' (is_tx={is_transcript})")

        # 1. Transform the query
        transformed_query = self._transform_query(query)
        if transformed_query != query:
            logger.info(f"Retriever: Using transformed query: '{transformed_query[:100]}...'")
        else:
            logger.info("Retriever: Using original query (transformation failed or unchanged).")

        logger.debug(f"Retriever: Attempting retrieve top {k}. Base ns: {self.namespace}, Event ID filter: {self.event_id}")

        try:
            # 2. Generate embedding for the (potentially transformed) query
            try:
                if not hasattr(self, 'embeddings'): raise RuntimeError("Embeddings missing")
                # Embed the transformed query
                query_embedding = self.embeddings.embed_query(transformed_query)
                logger.debug(f"Retriever: Query embedding generated (first 5): {query_embedding[:5]}...")
            except Exception as e: logger.error(f"Retriever: Embedding error: {e}", exc_info=True); return []

            # 3. Define namespaces & filters (no change here)
            namespaces = [self.namespace]
            event_ns = f"{self.namespace}-{self.event_id}" if self.event_id else None
            if event_ns and event_ns != self.namespace: namespaces.append(event_ns); logger.debug(f"Adding event ns: {event_ns}")

            # 4. Perform queries (no change here)
            all_matches = []
            for ns in namespaces:
                query_filter = {"agent_name": self.namespace}
                if self.event_id: query_filter["event_id"] = self.event_id; logger.debug(f"Adding event_id filter: {self.event_id}")
                else: logger.debug("No specific event_id filter applied.")
                logger.debug(f"Querying ns='{ns}', filter={query_filter}, top_k={k}")
                try:
                    response = self.index.query(vector=query_embedding, top_k=k, namespace=ns, filter=query_filter, include_metadata=True)
                    logger.debug(f"Raw response ns '{ns}': {response}")
                    if response.matches: logger.info(f"Found {len(response.matches)} matches in ns '{ns}'."); all_matches.extend(response.matches)
                    else: logger.info(f"No matches in ns '{ns}'.")
                except Exception as query_e: logger.error(f"Pinecone query error ns '{ns}': {query_e}", exc_info=True)

            if not all_matches: logger.warning("No matches found across namespaces."); return []

            # 5. Process & Rank with Time-Decay
            logger.debug(f"Total raw matches: {len(all_matches)}. Applying time-decay re-ranking...")
            decay_rate = 0.05 # Controls how quickly scores decay. Higher is faster.
            current_time = time.time()

            for match in all_matches:
                original_score = match.score
                if match.metadata:
                    is_core = match.metadata.get('is_core_memory', False)
                    saved_at = match.metadata.get('saved_at') # Expects a Unix timestamp

                    if is_core:
                        logger.info(f"Re-ranking ID {match.id}: Core memory, skipping decay. Score remains {original_score:.4f}")
                        continue

                    if saved_at:
                        try:
                            saved_at_ts = 0
                            if isinstance(saved_at, str):
                                # Clean up potential whitespace and newlines, then parse ISO 8601 format
                                saved_at_dt = datetime.fromisoformat(saved_at.strip())
                                # If the parsed datetime is naive, assume UTC. Otherwise, use its own timezone.
                                if saved_at_dt.tzinfo is None:
                                    saved_at_dt = saved_at_dt.replace(tzinfo=timezone.utc)
                                saved_at_ts = saved_at_dt.timestamp()
                            else:
                                # Handle cases where it might already be a Unix timestamp
                                saved_at_ts = float(saved_at)

                            age_seconds = current_time - saved_at_ts
                            age_days = age_seconds / (24 * 3600)
                            
                            if age_days > 0:
                                decay_factor = math.exp(-decay_rate * age_days)
                                match.score = original_score * decay_factor
                                logger.info(f"Re-ranking ID {match.id}: Original Score={original_score:.4f}, Age={age_days:.2f} days, New Score={match.score:.4f}")
                            else:
                                # Age is negative or zero, no decay
                                logger.info(f"Re-ranking ID {match.id}: Recent memory (age <= 0), no decay. Score remains {original_score:.4f}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not process 'saved_at' timestamp for match {match.id} ('{saved_at}'). Error: {e}. Skipping decay.")
                    else:
                        logger.info(f"Re-ranking ID {match.id}: No 'saved_at' timestamp. Skipping decay.")
                else:
                    logger.info(f"Re-ranking ID {match.id}: No metadata. Skipping decay.")

            # Sort again after applying time-decay
            all_matches.sort(key=lambda x: x.score, reverse=True)
            top_matches = all_matches[:k]

            logger.info(f"Top {len(top_matches)} matches after time-decay re-ranking:")
            for i, match in enumerate(top_matches):
                logger.info(f"  Rank {i+1}: ID={match.id}, Final Score={match.score:.4f}")


            # 6. Convert to Documents
            docs = []
            for match in top_matches:
                if not match.metadata: logger.warning(f"Match {match.id} lacks metadata."); continue
                content = match.metadata.get('content')
                if not content: logger.warning(f"Match {match.id} metadata lacks 'content'."); continue
                doc_metadata = {k: v for k, v in match.metadata.items() if k != 'content'}
                doc_metadata['score'] = match.score; doc_metadata['vector_id'] = match.id
                docs.append(Document(page_content=content, metadata=doc_metadata))

            logger.info(f"Retriever: Returning {len(docs)} processed contexts.")
            if docs: logger.debug(f"First doc metadata: {docs[0].metadata}")
            return docs

        except Exception as e: logger.error(f"Context retrieval error: {e}", exc_info=True); return []
