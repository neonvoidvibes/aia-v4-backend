from typing import Dict, List, Any, Optional, Protocol

class Chunk:
    """Represents a document chunk with metadata."""
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}

class VectorStore(Protocol):
    """Protocol for vector store implementations."""
    def search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Search the vector store and return relevant chunks."""
        ...
