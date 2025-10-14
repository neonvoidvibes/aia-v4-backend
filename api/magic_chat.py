from typing import Optional, Set
from core.retrieval.handler import RetrievalHandler
from core.retrieval.types import VectorStore

def _make_retriever(request, store: VectorStore) -> RetrievalHandler:
    """
    Create a RetrievalHandler with appropriate Tier-3 policy.

    Policy:
    - event_id "0000" (shared space): Tier-3 is ON by default (cross-event access)
    - All other events: Tier-3 is OFF by default (isolated to their own event)
    - Explicit allowlist from request overrides defaults

    Args:
        request: Request object with agent_name, event_id, and optional allowed_tier3_events
        store: VectorStore instance for retrieval

    Returns:
        RetrievalHandler configured with appropriate Tier-3 settings
    """
    agent_name = request.agent_name
    event_id = request.event_id

    # new: Tier-3 ON by default only for "0000"; OFF elsewhere unless explicitly enabled
    if event_id == "0000":
        rh = RetrievalHandler(store, agent_name, event_id, include_t3=True)
        # Optional: keep explicit allowlist if provided by client
        if hasattr(request, 'allowed_tier3_events') and request.allowed_tier3_events:
            rh.allowed_tier3_events = set(request.allowed_tier3_events)
    else:
        rh = RetrievalHandler(store, agent_name, event_id, include_t3=False, allowed_tier3_events=set())

    return rh
