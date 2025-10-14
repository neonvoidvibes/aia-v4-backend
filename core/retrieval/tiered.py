from typing import List, Dict, Optional, Set
from .types import Chunk, VectorStore
from .filters import merge_filters

def get_relevant_context_tiered(
    store: VectorStore,
    agent_name: str,
    event_id: str,
    query: str,
    # Policy: Tier-3 (cross-event) is ON by default only for shared space "0000",
    # otherwise OFF unless explicitly enabled by caller.
    include_t3: Optional[bool] = None,
    allowed_tier3_events: Optional[Set[str]] = None,
    k_per_tier: int = 8,
) -> List[Chunk]:
    base_filter = {"agent_name": agent_name, "event_id": event_id}
    # Prefer high-signal docs early
    t0_extra = {"content_category": "foundational_document"}
    t1_extra = {"content_category": {"$in": ["meeting_summary", "foundational_document"]}}
    t2_extra = {}

    # ---- Guardrails for Tier-3 defaults ----
    if include_t3 is None:
        include_t3 = (event_id == "0000")
    # If not explicitly provided and not in shared space, no cross-event reads.
    if allowed_tier3_events is None and event_id != "0000":
        allowed_tier3_events = set()

    # ----- T0/T1/T2: strict to current event -----
    # Examples (adjust to your tiers): foundational docs, summaries, recent chats
    results: List[Chunk] = []

    # T0: Foundational documents
    r = store.search(query, k=k_per_tier, metadata_filter=merge_filters(base_filter, t0_extra))
    results.extend(r)

    # T1: Meeting summaries and foundational docs
    r = store.search(query, k=k_per_tier, metadata_filter=merge_filters(base_filter, t1_extra))
    results.extend(r)

    # T2: General content from current event
    r = store.search(query, k=k_per_tier, metadata_filter=merge_filters(base_filter, t2_extra))
    results.extend(r)

    # ----- T3: cross-event expansion (optional) -----
    if include_t3:
        t3_filter: Dict = {"agent_name": agent_name}
        if allowed_tier3_events is not None:
            # Explicit allowlist requested by caller; empty set => no T3.
            if len(allowed_tier3_events) == 0:
                return results
            t3_filter["event_id"] = {"$in": list(allowed_tier3_events)}
        else:
            # Shared space (event_id=="0000") default: full namespace.
            # Do NOT apply $nin fallback anymore.
            if event_id != "0000":
                # Defensive: if a caller forced include_t3=True outside shared space
                # but forgot an allowlist, keep strict isolation.
                t3_filter["event_id"] = event_id
            # For "0000" we intentionally omit event_id to span all events for this agent.
        r = store.search(query, k=k_per_tier, metadata_filter=t3_filter)
        results.extend(r)

    return results
