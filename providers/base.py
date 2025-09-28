from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from core.types import ASRResult

class ASRProvider(ABC):
    name: str

    @abstractmethod
    def transcribe_file(self, wav_path: str, language: Optional[str]) -> ASRResult:
        ...
