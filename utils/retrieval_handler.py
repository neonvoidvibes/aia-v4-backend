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
DEFAULT_QUERY_TRANSFORM_PROMPT = """Rewrite the following user query to be more effective for searching a vector database. Your goal is to broaden the query slightly to catch related terms, but you must preserve the original keywords and intent.

**Guidelines:**
1.  **Preserve Core Keywords:** The most important keywords from the original query MUST be present in the rewritten query.
2.  **Simple is Better:** If the query is already specific and clear (e.g., asking about "tvättbjörn"), do not change it much. You might add synonyms if appropriate, but the original term is vital.
3.  **Extract Entities:** Identify and retain key entities (people, projects, dates).
4.  **No Hallucinations:** Do NOT add new topics or subjects that are not in the original query.

**Example 1:**
User Query: 'kan du se något om en tvättbjörn'
Rewritten Query: 'information about tvättbjörn raccoon'

**Example 2:**
User Query: 'what were the key decisions in the mobius project meeting on May 1st'
Rewritten Query: 'key decisions summary mobius project meeting May 1st'

Output only the rewritten query, no preamble.

User Query: '{user_query}'

Rewritten Query:"""


class RetrievalHandler:
    """Handles document retrieval using direct Pinecone queries."""

    def __init__(
        self,
        index_name: str = "river",
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        final_top_k: int = 10,
        initial_fetch_k: int = 100, # Fetch a larger pool for better re-ranking
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
        self.final_top_k = final_top_k
        self.initial_fetch_k = initial_fetch_k
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
                logger.info(f"Retriever: Connected to index '{self.index_name}'. Initial fetch k={self.initial_fetch_k}, Final k={self.final_top_k}")
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
            
        final_k = top_k or self.final_top_k
        initial_k = self.initial_fetch_k
        logger.debug(f"Retriever: Original query: '{query[:100]}...' (is_tx={is_transcript})")

        # 1. Transform the query
        transformed_query = self._transform_query(query)
        if transformed_query != query:
            logger.info(f"Retriever: Using transformed query: '{transformed_query[:100]}...'")
        else:
            logger.info("Retriever: Using original query (transformation failed or unchanged).")

        logger.debug(f"Retriever: Attempting retrieve top {initial_k} for re-ranking. Base ns: {self.namespace}, Event ID filter: {self.event_id}")

        try:
            # 2. Generate embedding for the (potentially transformed) query
            try:
                if not hasattr(self, 'embeddings'): raise RuntimeError("Embeddings missing")
                # Embed the transformed query
                query_embedding = self.embeddings.embed_query(transformed_query) # This line remains correct
                logger.debug(f"Retriever: Query embedding generated (first 5): {query_embedding[:5]}...")
            except Exception as e: logger.error(f"Retriever: Embedding error: {e}", exc_info=True); return []
    
            # 4. Perform query against the agent's namespace within the single shared index.
            all_matches = []
            query_filter = {"agent_name": self.namespace}
            if self.event_id:
                query_filter["event_id"] = self.event_id
            logger.debug(f"Querying index='{self.index_name}', ns='{self.namespace}', filter={query_filter}, top_k={initial_k}")
            try:
                response = self.index.query(vector=query_embedding, top_k=initial_k, namespace=self.namespace, filter=query_filter, include_metadata=True)
                if response.matches: all_matches.extend(response.matches)
            except Exception as query_e: logger.error(f"Pinecone query error for namespace '{self.namespace}': {query_e}", exc_info=True)

            if not all_matches: logger.warning("No matches found across namespaces."); return []

            # 5. Process & Rank with Time-Decay
            logger.debug(f"Total raw matches: {len(all_matches)}. Applying time-decay re-ranking...")
            decay_rate = 0.025 # Controls how quickly scores decay. Higher is faster.
            current_time = time.time()

            for match in all_matches:
                original_score = match.score
                if match.metadata:
                    is_core = match.metadata.get('is_core_memory', False)
                    created_at = match.metadata.get('created_at')
                    access_count = match.metadata.get('access_count', 0)
                    age_days = None

                    if is_core:
                        logger.info(f"Re-ranking ID {match.id}: Core memory, skipping decay. Score remains {original_score:.4f}")
                        match.metadata['age_days'] = "N/A (Core Memory)"
                        continue

                    if created_at:
                        try:
                            created_at_ts = 0
                            if isinstance(created_at, str):
                                # Handle ISO format strings with or without 'Z'
                                if created_at.endswith('Z'):
                                    created_at = created_at[:-1] + '+00:00'
                                created_at_dt = datetime.fromisoformat(created_at)
                                if created_at_dt.tzinfo is None:
                                    created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
                                created_at_ts = created_at_dt.timestamp()
                            else:
                                created_at_ts = float(created_at)

                            age_seconds = current_time - created_at_ts
                            age_days = age_seconds / (24 * 3600)
                            
                            if age_days < 1:
                                age_hours = age_seconds / 3600
                                match.metadata['age_display'] = f"{age_hours:.1f} hours"
                            else:
                                match.metadata['age_display'] = f"{age_days:.2f} days"

                            if age_days > 0:
                                # Reinforcement factor: Logarithmic growth to prevent runaway scores
                                # Adding 1 to access_count to avoid log(0) and ensure base is > 1
                                reinforcement_factor = math.log(1 + access_count)
                                
                                # The effective decay is reduced by how much the memory has been reinforced
                                # We add a small epsilon to avoid division by zero if reinforcement_factor is 0
                                effective_decay_rate = decay_rate / (1 + reinforcement_factor)
                                
                                decay_factor = math.exp(-effective_decay_rate * age_days)
                                match.score = original_score * decay_factor
                                logger.info(f"Re-ranking ID {match.id}: Original Score={original_score:.4f}, Age={match.metadata['age_display']}, Accesses={access_count}, New Score={match.score:.4f}")
                            else:
                                logger.info(f"Re-ranking ID {match.id}: Recent memory (age <= 0), no decay. Score remains {original_score:.4f}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not process 'created_at' timestamp for match {match.id} ('{created_at}'). Error: {e}. Skipping decay.")
                            match.metadata['age_display'] = "Unknown"
                    else:
                        logger.info(f"Re-ranking ID {match.id}: No 'created_at' timestamp. Skipping decay.")
                        match.metadata['age_display'] = "Unknown"
                else:
                    logger.info(f"Re-ranking ID {match.id}: No metadata. Skipping decay.")
                    if match.metadata is None: match.metadata = {}
                    match.metadata['age_display'] = "Unknown"

            # Sort again after applying time-decay
            all_matches.sort(key=lambda x: x.score, reverse=True)
            top_matches = all_matches[:final_k]

            logger.info(f"Top {len(top_matches)} matches after time-decay re-ranking:")
            for i, match in enumerate(top_matches):
                logger.info(f"  Rank {i+1}: ID={match.id}, Final Score={match.score:.4f}")


            # 6. Convert to Documents
            docs = []
            for match in top_matches:
                if not match.metadata: logger.warning(f"Match {match.id} lacks metadata."); continue
                content = match.metadata.get('content')
                if not content: logger.warning(f"Match {match.id} metadata lacks 'content'."); continue
                # Ensure age_display is set for core memories if it wasn't before
                if match.metadata.get('is_core_memory'):
                    match.metadata['age_display'] = "N/A (Core Memory)"
                doc_metadata = {k: v for k, v in match.metadata.items() if k != 'content'}
                doc_metadata['score'] = match.score; doc_metadata['vector_id'] = match.id
                docs.append(Document(page_content=content, metadata=doc_metadata))

            logger.info(f"Retriever: Returning {len(docs)} processed contexts.")
            if docs: logger.debug(f"First doc metadata: {docs[0].metadata}")
            return docs

        except Exception as e: logger.error(f"Context retrieval error: {e}", exc_info=True); return []

    def reinforce_memories(self, docs: List[Document]):
        """Increments the access_count for a list of documents."""
        if not self.index:
            logger.warning("Cannot reinforce memories: Pinecone index not available.")
            return

        logger.info(f"Reinforcing {len(docs)} memories...")
        for doc in docs:
            try:
                vector_id = doc.metadata.get('vector_id')
                if not vector_id:
                    logger.warning(f"Cannot reinforce document, missing 'vector_id' in metadata: {doc.metadata}")
                    continue

                # Fetch the latest metadata to get the current access_count
                try:
                    fetch_response = self.index.fetch(ids=[vector_id], namespace=self.namespace)
                    if not fetch_response.vectors:
                        logger.warning(f"Could not fetch vector {vector_id} for reinforcement.")
                        continue
                    
                    current_metadata = fetch_response.vectors[vector_id].metadata
                    current_count = current_metadata.get('access_count', 0)
                except Exception as fetch_e:
                    logger.error(f"Failed to fetch metadata for reinforcement of {vector_id}: {fetch_e}")
                    continue # Skip if we can't be sure of the current count

                new_count = current_count + 1
                
                # Update the metadata for the vector
                self.index.update(
                    id=vector_id,
                    set_metadata={'access_count': new_count},
                    namespace=self.namespace
                )
                logger.debug(f"Reinforced memory {vector_id}, new access_count: {new_count}")

            except Exception as e:
                logger.error(f"Failed to reinforce memory for doc: {doc.metadata.get('vector_id', 'N/A')}. Error: {e}", exc_info=True)
        logger.info("Memory reinforcement complete.")
