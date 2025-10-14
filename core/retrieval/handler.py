from typing import List, Optional, Set
from .types import Chunk, VectorStore
from .tiered import get_relevant_context_tiered

class RetrievalHandler:
    """Handler for document retrieval with tiered context."""

    def __init__(self, store: VectorStore, agent_name: str, event_id: str,
                 include_t3: Optional[bool] = None, allowed_tier3_events: Optional[Set[str]] = None):
        self.store = store
        self.agent_name = agent_name
        self.event_id = event_id
        # Default Tier-3 policy mirrors tiered.get_relevant_context_tiered
        self.include_t3 = include_t3 if include_t3 is not None else (event_id == "0000")
        self.allowed_tier3_events = allowed_tier3_events

    def get_context(
        self,
        query: str,
        k_per_tier: int = 8
    ) -> List[Chunk]:
        """
        Retrieve relevant context using tiered retrieval strategy.

        Args:
            query: The search query
            k_per_tier: Number of results to retrieve per tier

        Returns:
            List of relevant document chunks
        """
        return get_relevant_context_tiered(
            store=self.store,
            agent_name=self.agent_name,
            event_id=self.event_id,
            query=query,
            include_t3=self.include_t3,
            allowed_tier3_events=self.allowed_tier3_events,
            k_per_tier=k_per_tier,
        )
