"""Utilities for retrieving relevant document context for chat."""
import logging
import threading
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
from utils.pinecone_utils import init_pinecone, get_index
from anthropic import Anthropic # Import Anthropic client

logger = logging.getLogger(__name__)

# Thread-safe, module-scope embeddings cache keyed by (api_key or 'ENV', model)
_EMBEDDINGS_CACHE = {}
_EMBEDDINGS_LOCK = threading.Lock()

def get_cached_embeddings(model_name: str, api_key: Optional[str]) -> OpenAIEmbeddings:
    key = (api_key or "ENV", model_name)
    emb = _EMBEDDINGS_CACHE.get(key)
    if emb is not None:
        return emb
    with _EMBEDDINGS_LOCK:
        emb = _EMBEDDINGS_CACHE.get(key)
        if emb is None:
            emb = OpenAIEmbeddings(model=model_name, api_key=api_key)
            _EMBEDDINGS_CACHE[key] = emb
            logger.info(f"Retriever: Created Embeddings model '{model_name}' and cached (keyed by API).")
    return emb

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
        initial_fetch_k: int = 50, # Increased initial pool per request
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None # Agent-specific key
    ):
        """Initialize retrieval handler."""
        if not agent_name: raise ValueError("agent_name required")
        if not anthropic_api_key: raise ValueError("anthropic_api_key required for query transformation")

        self.index_name = index_name
        self.namespace = agent_name
        self.session_id = session_id
        self.event_id = event_id if event_id and event_id != '0000' else None
        self.final_top_k = final_top_k
        self.initial_fetch_k = initial_fetch_k
        self.embedding_model_name = "text-embedding-3-small"
        self.anthropic_api_key = anthropic_api_key

        try:
            self.embeddings = get_cached_embeddings(self.embedding_model_name, openai_api_key)
        except Exception as e:
            raise RuntimeError("Failed Embeddings init") from e
        
        try:
            index = get_index(self.index_name)
            if not index:
                logger.warning(f"Retriever: Index '{self.index_name}' does not exist. RAG will be disabled for this handler.")
                self.index = None
            else:
                self.index = index
                logger.info(f"Retriever: Connected to index '{self.index_name}'. Initial fetch k={self.initial_fetch_k}, Final k={self.final_top_k}")
        except Exception as e:
            logger.error(f"Retriever: Error connecting to or checking index '{self.index_name}': {e}", exc_info=True)
            self.index = None # Ensure index is None on any error

    def _transform_query(self, query: str) -> str:
        """Uses LLM to rewrite the query for better vector search."""
        logger.debug(f"Transforming query: '{query}'")
        try:
            # NEW: Create a transient client with the correct agent-specific key
            transient_client = Anthropic(api_key=self.anthropic_api_key)

            prompt = DEFAULT_QUERY_TRANSFORM_PROMPT.format(user_query=query)

            # Use the new transient client
            message = transient_client.messages.create(
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
                response = self.index.query(
                    vector=query_embedding,
                    top_k=initial_k,
                    namespace=self.namespace,
                    filter=query_filter,
                    include_metadata=True,
                    include_values=False,
                )
                if response.matches: all_matches.extend(response.matches)
            except Exception as query_e: logger.error(f"Pinecone query error for namespace '{self.namespace}': {query_e}", exc_info=True)

            if not all_matches: logger.warning("No matches found across namespaces."); return []

            # 5. Process & Rank with Time-Decay
            logger.debug(f"Total raw matches: {len(all_matches)}. Applying time-decay re-ranking...")
            decay_rate = 0.015 # Controls how quickly scores decay. Higher is faster. (Reduced from 0.025)
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
                                decayed_score = original_score * decay_factor

                                # Implement a score floor for very high-quality original matches to prevent them from decaying too much.
                                score_floor = 0.72
                                if original_score > 0.85 and decayed_score < score_floor:
                                    match.score = score_floor
                                    logger.info(f"Re-ranking ID {match.id}: High-quality match, applying score floor. Original: {original_score:.4f}, Decayed: {decayed_score:.4f}, Floored: {match.score:.4f}")
                                else:
                                    match.score = decayed_score

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

    def _mmr(self, candidates, embeddings, k: int = 15, lambda_mult: float = 0.6):
        """Simple MMR over candidate list using score as relevance and cosine similarity over embeddings.
        candidates: list of Pinecone match objects with .score and .id
        embeddings: dict id->embedding vector (same query embedding used to derive similarity). If not provided,
        fallback to score-only ranking.
        """
        try:
            import numpy as np
        except Exception:
            # Fallback: return top-k by original score
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:k]

        if not candidates:
            return []
        selected = []
        remaining = candidates[:]
        # Normalize score to [0,1]
        max_s = max([c.score for c in candidates]) or 1.0
        min_s = min([c.score for c in candidates])
        denom = (max_s - min_s) or 1.0

        def cosine(a, b):
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na == 0 or nb == 0: return 0.0
            return float(np.dot(a, b) / (na * nb))

        while remaining and len(selected) < k:
            best = None; best_score = -1e9
            for cand in remaining:
                rel = (cand.score - min_s) / denom
                if not embeddings or cand.id not in embeddings:
                    mmr_score = rel
                else:
                    # diversity term: 1 - max cosine to already selected
                    if not selected:
                        div = 1.0
                    else:
                        cemb = embeddings.get(cand.id)
                        max_sim = max([cosine(cemb, embeddings.get(s.id, cemb)) for s in selected])
                        div = 1.0 - max_sim
                    mmr_score = lambda_mult * rel + (1 - lambda_mult) * div
                if mmr_score > best_score:
                    best = cand; best_score = mmr_score
            selected.append(best)
            remaining.remove(best)
        return selected

    def get_relevant_context_tiered(
        self,
        query: str,
        tier_caps: List[int] = [7, 5, 3],
        mmr_k: int = 15,
        mmr_lambda: float = 0.6,
        include_t3: bool = True,
    ) -> List[Document]:
        """Event-scoped retrieval across tiers with MMR over union.
        Tier1: {agent_name, event_id=current}
        Tier2: {agent_name, event_id='0000'}
        Tier3: {agent_name, event_id!=current and !='0000'} (optional)
        """
        if not self.index:
            logger.info("Retriever: Skipping context retrieval as Pinecone index is not available.")
            return []

        # Prepare embedding for query (transformed)
        transformed_query = self._transform_query(query)
        try:
            query_embedding = self.embeddings.embed_query(transformed_query)
        except Exception as e:
            logger.error(f"Retriever tiered: Embedding error: {e}", exc_info=True)
            return []

        tier_results = []
        tiers = []
        current_event = self.event_id or '0000'
        # Tier 1
        tiers.append({"filter": {"agent_name": self.namespace, "event_id": current_event}, "cap": tier_caps[0] if len(tier_caps)>0 else 7, "label": "t1"})
        # Tier 2
        tiers.append({"filter": {"agent_name": self.namespace, "event_id": "0000"}, "cap": tier_caps[1] if len(tier_caps)>1 else 5, "label": "t2"})
        # Tier 3
        if include_t3:
            tiers.append({"filter": {"agent_name": self.namespace, "event_id": {"$ne": current_event}}, "cap": tier_caps[2] if len(tier_caps)>2 else 3, "label": "t3"})

        all_matches = []
        tier_hit_counts = {"t1": 0, "t2": 0, "t3": 0}
        for tier in tiers:
            try:
                resp = self.index.query(
                    vector=query_embedding,
                    top_k=tier["cap"],
                    namespace=self.namespace,
                    filter=tier["filter"],
                    include_metadata=True,
                    include_values=False,
                )
                matches = resp.matches or []
                tier_hit_counts[tier["label"]] = len(matches)
                all_matches.extend(matches)
            except Exception as e:
                logger.error(f"Tiered query error ({tier['label']}): {e}")

        if not all_matches:
            logger.info("Retriever tiered: 0 results across tiers")
            return []

        # Build id->embedding cache if index supports; otherwise, use None and fallback to scores
        # Pinecone query does not return embeddings; skip deep diversity across text, use score-only fallback in _mmr
        selected = self._mmr(all_matches, embeddings=None, k=mmr_k, lambda_mult=mmr_lambda)

        # Convert to docs
        docs: List[Document] = []
        for m in selected:
            if not m.metadata: continue
            content = m.metadata.get('content')
            if not content: continue
            meta = {k: v for (k, v) in m.metadata.items() if k != 'content'}
            # Label non-event sources inline as requested
            src_event = str(meta.get('event_id', '0000'))
            if current_event and src_event != current_event:
                if src_event == '0000':
                    meta['source_label'] = '[shared-0000]'
                else:
                    meta['source_label'] = f"[other:{src_event}]"
            meta['score'] = m.score
            meta['vector_id'] = m.id
            docs.append(Document(page_content=content, metadata=meta))

        logger.info(f"Retriever tiered: hits t1={tier_hit_counts['t1']} t2={tier_hit_counts['t2']} t3={tier_hit_counts['t3']}; returning {len(docs)}")
        return docs

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
