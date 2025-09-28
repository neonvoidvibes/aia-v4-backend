from __future__ import annotations
from typing import Tuple
from core.types import ASRResult, AdvisoryFlags

# Non-destructive advisory checks. Never mutate text to empty. Never block append.
# Cheap near-dup check using normalized text hash on a short window.

class NearDupCache:
    def __init__(self, max_items: int = 50) -> None:
        self._max = max_items
        self._queue: list[tuple[str, int]] = []   # (hash, seq)
        self._set: set[str] = set()

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(s.lower().strip().split())

    def check(self, text: str, seq: int) -> bool:
        h = hash(self._norm(text))
        if str(h) in self._set:
            return True
        # enqueue
        self._queue.append((str(h), seq))
        self._set.add(str(h))
        if len(self._queue) > self._max:
            old, _ = self._queue.pop(0)
            self._set.discard(old)
        return False

def advisory_flags(asr: ASRResult, seq: int, dup_cache: NearDupCache) -> AdvisoryFlags:
    is_dup = dup_cache.check(asr.raw_text, seq)
    return AdvisoryFlags(is_near_dup=is_dup, notes="near-duplicate" if is_dup else None)

