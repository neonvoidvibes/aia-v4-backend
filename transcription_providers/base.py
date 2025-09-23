from __future__ import annotations
from typing import Any, Dict, List, Optional

Segment = Dict[str, Any]
TranscriptionResult = Dict[str, Any]

class TranscriptionProvider:
    def transcribe_file(
        self,
        path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Optional[TranscriptionResult]:
        raise NotImplementedError