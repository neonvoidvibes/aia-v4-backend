"""Utilities for retrieving relevant document context for chat."""
import os
import logging
import threading
import traceback
import time
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Iterable, Set

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
DEFAULT_QUERY_TRANSFORM_PROMPT = """Rewrite the following user query to be more effective for searching a vector database. Your goal is to expand the query with related terms and synonyms while preserving the original keywords and intent.

**Guidelines:**
1.  **Preserve Core Keywords:** The most important keywords from the original query MUST be present in the rewritten query.
2.  **Add Synonyms & Related Terms:** Include formal/informal variations, technical terms, and conceptually related words.
3.  **Match Document Language:** Consider both colloquial user language and formal document language that might contain the same concepts.
4.  **Extract Entities:** Identify and retain key entities (people, projects, dates, organizations).
5.  **No Hallucinations:** Do NOT add new topics or subjects that are not in the original query.

**Examples:**
User Query: 'kan du se något om en tvättbjörn'
Rewritten Query: information about tvättbjörn raccoon

User Query: 'what were the key decisions in the mobius project meeting on May 1st'
Rewritten Query: key decisions summary mobius project meeting May 1st

User Query: 'company rules'
Rewritten Query: company rules policies guidelines regulations procedures standards

User Query: 'team structure'
Rewritten Query: team structure organization hierarchy roles responsibilities management

Output only the rewritten query, no preamble or formatting.

User Query: '{user_query}'"""


class RetrievalHandler:
    """Handles document retrieval using direct Pinecone queries."""

    def __init__(
        self,
        index_name: str = "river",
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        final_top_k: int = 24,
        initial_fetch_k: int = 120, # Increased initial pool per request
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None, # Agent-specific key
        event_type: Optional[str] = None,
        personal_event_id: Optional[str] = None,
        allowed_tier3_events: Optional[Iterable[str]] = None,
        include_personal_tier: bool = False,
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
        # Use environment variable or default to text-embedding-3-small
        self.embedding_model_name = os.getenv('RETRIEVAL_EMBED_MODEL', 'text-embedding-3-small')
        self.anthropic_api_key = anthropic_api_key
        self.personal_event_id = personal_event_id if personal_event_id and personal_event_id != '0000' else None
        self.include_personal_tier = bool(include_personal_tier and self.personal_event_id)
        inferred_event_type = (event_type or "").lower()
        if not inferred_event_type:
            if self.event_id and self.personal_event_id and self.event_id == self.personal_event_id:
                inferred_event_type = 'personal'
            elif self.event_id:
                inferred_event_type = 'group'
            else:
                inferred_event_type = 'shared'
        self.event_type = inferred_event_type
        self.allowed_tier3_events: Optional[Set[str]]
        if allowed_tier3_events is None:
            self.allowed_tier3_events = None
        else:
            self.allowed_tier3_events = {ev for ev in allowed_tier3_events if ev and ev != '0000'}
        self.default_tier_caps: Dict[str, int] = {
            't0_foundation': 4,
            't_personal': 12,
            't1': 6,
            't2': 6,
            't3': 4,
        }
        self._last_retrieval_breakdown: Dict[str, List[Document]] = {}
        self._last_tier_hit_counts: Dict[str, int] = {}

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
            raw_output = message.content[0].text.strip()
            
            # Clean up the output - remove any formatting artifacts from the LLM response
            transformed_query = raw_output
            # Remove common prefixes the LLM might add
            for prefix in ["Rewritten Query:", "Rewritten query:", "Query:", "Transformed query:", "Transformed Query:"]:
                if transformed_query.startswith(prefix):
                    transformed_query = transformed_query[len(prefix):].strip()
            
            # Remove quotes if the entire query is wrapped in quotes
            if transformed_query.startswith('"') and transformed_query.endswith('"'):
                transformed_query = transformed_query[1:-1].strip()
            if transformed_query.startswith("'") and transformed_query.endswith("'"):
                transformed_query = transformed_query[1:-1].strip()
                
            logger.debug(f"Raw LLM output: '{raw_output[:100]}...'")
            logger.debug(f"Cleaned transformed query: '{transformed_query}'")
            
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

    def _build_diversity_cache(self, matches) -> Dict[str, List[float]]:
        """Build diversity vectors based on document metadata for MMR calculation.
        Creates pseudo-embeddings that represent document characteristics for diversity."""
        try:
            import numpy as np
        except ImportError:
            logger.warning("NumPy not available, MMR will use score-only fallback")
            return {}
        
        diversity_cache = {}
        
        for match in matches:
            if not match.metadata:
                continue
                
            # Create diversity vector based on document characteristics
            diversity_vector = []
            
            # Content type diversity (foundational vs chat vs other)
            content_cat = match.metadata.get('content_category', 'unknown')
            if content_cat == 'foundational_document':
                diversity_vector.extend([1.0, 0.0, 0.0])
            elif content_cat == 'chat_message':
                diversity_vector.extend([0.0, 1.0, 0.0])
            else:
                diversity_vector.extend([0.0, 0.0, 1.0])
                
            # Event diversity
            event_id = str(match.metadata.get('event_id', '0000'))
            if event_id == '0000':
                diversity_vector.extend([1.0, 0.0])
            else:
                diversity_vector.extend([0.0, 1.0])
                
            # Source diversity (filename similarity)
            filename = match.metadata.get('filename', 'unknown')
            filename_hash = hash(filename) % 100  # Simple filename fingerprint
            diversity_vector.append(filename_hash / 100.0)
            
            # Temporal diversity (recent vs old)
            access_count = match.metadata.get('access_count', 0)
            is_core = match.metadata.get('is_core_memory', False)
            if is_core:
                diversity_vector.extend([1.0, 0.0])
            elif access_count > 10:
                diversity_vector.extend([0.0, 1.0]) 
            else:
                diversity_vector.extend([0.0, 0.0])
                
            # Pad to consistent length
            while len(diversity_vector) < 8:
                diversity_vector.append(0.0)
                
            diversity_cache[match.id] = diversity_vector
            
        logger.debug(f"Built diversity cache for {len(diversity_cache)} documents")
        return diversity_cache

    def _boost_filename_matches(self, matches: List, query: str) -> List:
        """Boost scores for documents where query terms match filename/doc_id."""
        if not matches:
            return matches
            
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        boosted = []
        for match in matches:
            filename = match.metadata.get('file_name', '').lower()
            doc_id = match.metadata.get('doc_id', '').lower()
            
            # Calculate boost based on term matches in filename/doc_id
            boost = 0.0
            for term in query_terms:
                if len(term) > 2:  # Skip short words
                    if term in filename or term in doc_id:
                        boost += 0.1  # Add 0.1 boost per matching term
            
            # Apply boost to score by modifying the match in-place
            if boost > 0:
                original_score = match.score
                match.score = match.score + boost
                boosted.append(match)
                logger.info(f"Boosted {filename or doc_id} by {boost:.2f} (new score: {match.score:.4f})")
            else:
                boosted.append(match)
                
        # Re-sort by score after boosting
        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    def _resolve_tier_caps(self, override: Optional[Iterable[int]]) -> Dict[str, int]:
        caps = dict(self.default_tier_caps)
        if override is None:
            return caps

        if isinstance(override, dict):
            for key, value in override.items():
                if key in caps:
                    try:
                        caps[key] = max(int(value), 0)
                    except (TypeError, ValueError):
                        continue
            return caps

        try:
            iterable_values = list(override)
        except TypeError:
            return caps

        if len(iterable_values) <= 4:
            legacy_order = ['t0_foundation', 't1', 't2', 't3']
            for idx, value in enumerate(iterable_values):
                if idx >= len(legacy_order):
                    break
                try:
                    caps[legacy_order[idx]] = max(int(value), 0)
                except (TypeError, ValueError):
                    continue
            return caps

        mapping_order = ['t0_foundation', 't_personal', 't1', 't2', 't3']
        for idx, value in enumerate(iterable_values):
            if idx >= len(mapping_order):
                break
            try:
                caps[mapping_order[idx]] = max(int(value), 0)
            except (TypeError, ValueError):
                continue
        return caps

    def get_relevant_context_tiered(
        self,
        query: str,
        tier_caps: Optional[Iterable[int]] = None,
        mmr_k: Optional[int] = None,
        mmr_lambda: float = 0.6,
        include_t3: bool = True,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Document]:
        """Tiered retrieval that prioritizes personal context when available."""
        if not self.index:
            logger.info("Retriever: Skipping context retrieval as Pinecone index is not available.")
            self._last_retrieval_breakdown = {}
            self._last_tier_hit_counts = {}
            return []

        transformed_query = self._transform_query(query)
        try:
            query_embedding = self.embeddings.embed_query(transformed_query)
        except Exception as e:
            logger.error(f"Retriever tiered: Embedding error: {e}", exc_info=True)
            self._last_retrieval_breakdown = {}
            self._last_tier_hit_counts = {}
            return []

        current_event = self.event_id or '0000'
        caps = self._resolve_tier_caps(tier_caps)
        tiers: List[Dict[str, Any]] = []
        tier_hit_counts = {label: 0 for label in ['t0_foundation', 't_personal', 't1', 't2', 't3']}

        def merge_filter(base: Dict[str, Any]) -> Dict[str, Any]:
            if not metadata_filter:
                return base
            merged = dict(base)
            for key, value in metadata_filter.items():
                if key not in ('agent_name', 'event_id'):
                    merged[key] = value
            return merged

        if caps['t0_foundation'] > 0:
            tier0_filter = merge_filter({'agent_name': self.namespace, 'content_category': 'foundational_document'})
            tiers.append({'filter': tier0_filter, 'cap': caps['t0_foundation'], 'label': 't0_foundation'})

        if self.include_personal_tier and caps['t_personal'] > 0:
            personal_filter = merge_filter({'agent_name': self.namespace, 'event_id': self.personal_event_id})
            tiers.append({'filter': personal_filter, 'cap': caps['t_personal'], 'label': 't_personal'})

        if caps['t1'] > 0:
            t1_filter = merge_filter({'agent_name': self.namespace, 'event_id': current_event})
            tiers.append({'filter': t1_filter, 'cap': caps['t1'], 'label': 't1'})

        if caps['t2'] > 0:
            t2_filter = merge_filter({'agent_name': self.namespace, 'event_id': '0000'})
            tiers.append({'filter': t2_filter, 'cap': caps['t2'], 'label': 't2'})

        if include_t3 and caps['t3'] > 0:
            if self.allowed_tier3_events is not None:
                if self.allowed_tier3_events:
                    t3_event_clause: Optional[Dict[str, Any]] = {'$in': list(self.allowed_tier3_events)}
                else:
                    t3_event_clause = None
            else:
                t3_event_clause = {'$nin': [current_event, '0000']}

            if t3_event_clause:
                t3_filter = merge_filter({'agent_name': self.namespace, 'event_id': t3_event_clause})
                tiers.append({'filter': t3_filter, 'cap': caps['t3'], 'label': 't3'})

        all_matches = []
        for tier in tiers:
            cap = tier['cap']
            if not cap:
                continue
            tier_filter = tier['filter']
            if isinstance(tier_filter.get('event_id'), dict):
                clause = tier_filter['event_id']
                if clause.get('$in') == []:
                    continue
            try:
                resp = self.index.query(
                    vector=query_embedding,
                    top_k=cap,
                    namespace=self.namespace,
                    filter=tier_filter,
                    include_metadata=True,
                    include_values=False,
                )
                matches = resp.matches or []
                tier_hit_counts[tier['label']] = len(matches)
                for match in matches:
                    match.metadata = match.metadata or {}
                    tiers_list = match.metadata.get('retrieval_tiers') or []
                    tiers_list.append(tier['label'])
                    match.metadata['retrieval_tiers'] = tiers_list
                    if tier['label'] == 't_personal' or 'retrieval_tier' not in match.metadata:
                        match.metadata['retrieval_tier'] = tier['label']
                all_matches.extend(matches)
            except Exception as e:
                logger.error(f"Tiered query error ({tier['label']}): {e}")

        if not all_matches:
            logger.info("Retriever tiered: 0 results across tiers")
            self._last_retrieval_breakdown = {}
            self._last_tier_hit_counts = tier_hit_counts
            return []

        unique_matches: Dict[str, Any] = {}
        for match in all_matches:
            match.metadata = match.metadata or {}
            tiers_list = list(dict.fromkeys(match.metadata.get('retrieval_tiers') or []))
            match.metadata['retrieval_tiers'] = tiers_list
            existing = unique_matches.get(match.id)
            if not existing:
                unique_matches[match.id] = match
                continue

            existing.metadata = existing.metadata or {}
            existing_tiers = list(dict.fromkeys(existing.metadata.get('retrieval_tiers') or []))
            combined_tiers = list(dict.fromkeys(existing_tiers + tiers_list))

            preferred = existing if existing.score >= match.score else match
            other = match if preferred is existing else existing
            preferred.metadata = preferred.metadata or {}
            preferred.metadata['retrieval_tiers'] = combined_tiers
            if 't_personal' in combined_tiers:
                preferred.metadata['retrieval_tier'] = 't_personal'
            else:
                preferred.metadata.setdefault('retrieval_tier', combined_tiers[0] if combined_tiers else None)

            unique_matches[match.id] = preferred

        for match in unique_matches.values():
            tiers_list = match.metadata.get('retrieval_tiers') or []
            if 't_personal' in tiers_list:
                match.metadata['retrieval_tier'] = 't_personal'
            elif tiers_list and match.metadata.get('retrieval_tier') not in tiers_list:
                match.metadata['retrieval_tier'] = tiers_list[0]

        deduped_matches = list(unique_matches.values())
        if not deduped_matches:
            logger.info("Retriever tiered: matches removed during deduplication")
            self._last_retrieval_breakdown = {}
            self._last_tier_hit_counts = tier_hit_counts
            return []

        logger.info(f"About to boost {len(deduped_matches)} matches for query: '{query}'")
        boosted_matches = self._boost_filename_matches(deduped_matches, query)
        logger.info(f"Boost completed, candidates={len(boosted_matches)}")

        diversity_cache = self._build_diversity_cache(boosted_matches)
        mmr_target = mmr_k or max(self.final_top_k + 8, 32)
        selected = self._mmr(boosted_matches, embeddings=diversity_cache, k=mmr_target, lambda_mult=mmr_lambda)

        if self.final_top_k and len(selected) > self.final_top_k:
            selected = selected[:self.final_top_k]

        docs: List[Document] = []
        breakdown: Dict[str, List[Document]] = {}
        for match in selected:
            metadata = match.metadata or {}
            content = metadata.get('content')
            if not content:
                continue

            meta = {k: v for (k, v) in metadata.items() if k != 'content'}
            tier_label = meta.get('retrieval_tier')
            src_event = str(meta.get('event_id', '0000'))

            if tier_label == 't_personal':
                meta['source_label'] = f"[personal:{src_event}]"
            elif current_event and src_event != current_event:
                if src_event == '0000':
                    meta['source_label'] = '[shared-0000]'
                else:
                    meta['source_label'] = f"[other:{src_event}]"

            meta['score'] = match.score
            meta['vector_id'] = match.id

            doc = Document(page_content=content, metadata=meta)
            docs.append(doc)
            breakdown.setdefault(tier_label or 'unknown', []).append(doc)

        self._last_retrieval_breakdown = breakdown
        self._last_tier_hit_counts = tier_hit_counts
        logger.info("Retriever tiered: hits %s; returning %d documents", tier_hit_counts, len(docs))
        return docs



    def get_last_retrieval_breakdown(self) -> Dict[str, List[Document]]:
        return self._last_retrieval_breakdown

    def get_last_tier_hit_counts(self) -> Dict[str, int]:
        return self._last_tier_hit_counts


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
