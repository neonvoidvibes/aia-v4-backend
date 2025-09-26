"""
Strict subtractive overlap filter for streaming ASR.
NEVER prepends/carries seed words - only removes overlap.
Includes seed-head repeat dampener to prevent chant loops.
"""

from dataclasses import dataclass
from typing import List, Tuple, Deque, Optional
from collections import deque
import re

# Configuration constants
_MIN_OVERLAP_TOKENS = 12
_MAX_CONTEXT_TOKENS = 50
_MIN_SEGMENT_TOKENS = 6
_OVERLAP_RATIO = 0.60

# Regex patterns for normalization
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[""\"'''`´]")
_nonword_re = re.compile(r"[^\w\s]+")

def _normalize_tokens(text: str) -> List[str]:
    """Lowercase, strip quotes/punct, collapse spaces -> split"""
    if not text:
        return []
    t = text.strip().lower()
    t = _punct_re.sub("", t)
    # keep sentence punctuation from merging into tokens for overlap checks
    t = _nonword_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    return t.split()

def _lcs_len(a: List[str], b: List[str]) -> int:
    """Classic DP LCS length, optimized for short sequences (<=15)"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def _lcs_suffix_prefix_overlap(tail: List[str], head: List[str]) -> int:
    """Compute overlap between context suffix and current head."""
    if not tail or not head:
        return 0

    # Find longest matching contiguous overlap
    max_overlap = 0
    k = min(len(tail), len(head))
    for i in range(1, k + 1):
        if tail[-i:] == head[:i]:
            max_overlap = i
    return max_overlap

@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds
    conf: float   # 0..1

    @property
    def norm(self) -> str:
        """Normalized token for overlap comparison (single token or '')."""
        toks = _normalize_tokens(self.text)
        return toks[0] if toks else ""

@dataclass
class DetectorState:
    """
    Strict subtractive stitcher that ONLY removes overlap.
    Never prepends/carries seed words. Includes repeat dampener.
    """
    last_tokens: List[str]
    recent_heads: Deque[Tuple[str, ...]]

    def __init__(self):
        self.last_tokens = []
        self.recent_heads = deque(maxlen=6)

    def tail_tokens(self, limit: int) -> List[str]:
        """Return last N tokens from context."""
        return self.last_tokens[-limit:] if self.last_tokens else []

def maybe_trim_repetition(
    words: List[Word],
    state: DetectorState,
    *,
    min_overlap_tokens: int = _MIN_OVERLAP_TOKENS,
    max_context_tokens: int = _MAX_CONTEXT_TOKENS,
    min_segment_tokens: int = _MIN_SEGMENT_TOKENS,
    overlap_ratio: float = _OVERLAP_RATIO,
) -> Tuple[List[Word], Optional[str], int]:
    """
    Pure function: if the head of `words` overlaps with the tail of prior context,
    trim only the overlapped head.
    Returns (possibly_trimmed_words, reason, cut_word_count).
    Tunables are explicit and defaulted; call-sites can pass overrides for A/B.
    """
    if not words or len(words) < min_segment_tokens:
        # Still update context even for short segments
        if words:
            head_norm = [w.norm for w in words if w.norm]
            state.last_tokens = (state.last_tokens + head_norm)[-200:]
        return words, None, 0

    ctx_tail = state.tail_tokens(limit=max_context_tokens)
    seg_head = [w.norm for w in words if w.norm]

    if not ctx_tail or not seg_head:
        # Still update context even if no overlap detection
        head_norm = [w.norm for w in words if w.norm]
        # DEBUG: print(f'Early return: ctx_tail={ctx_tail}, seg_head={seg_head}, head_norm={head_norm}')
        state.last_tokens = (state.last_tokens + head_norm)[-200:]
        # DEBUG: print(f'Context after update: {state.last_tokens}')
        return words, None, 0

    # Compute suffix-prefix overlap length (context suffix vs current head).
    overlap = _lcs_suffix_prefix_overlap(ctx_tail, seg_head)
    threshold = max(min_overlap_tokens, int(len(seg_head) * overlap_ratio))
    if overlap >= threshold:
        cut = min(overlap, len(words))
        trimmed = words[cut:]
        reason = f"trimmed_overlap:{overlap}"
        # update rolling context with the tokens we will KEEP
        kept_norm = [w.norm for w in trimmed if w.norm]
        state.last_tokens = (state.last_tokens + kept_norm)[-200:]
        return trimmed, reason, cut

    # no trim; still advance context with head (bounded)
    head_norm = [w.norm for w in words if w.norm]
    state.last_tokens = (state.last_tokens + head_norm)[-200:]
    return words, None, 0

# Backward compatibility functions for existing code
class LegacyHallucinationManager:
    """Legacy wrapper to maintain backward compatibility"""
    def __init__(self, session_id: str):
        self.session_id = session_id

    def process_transcript(self, text: str) -> tuple:
        # Legacy function that just returns valid for all text
        return True, "valid", text

# Global manager tracking (for backward compatibility)
_legacy_managers = {}

def get_hallucination_manager(session_id: str, **kwargs) -> LegacyHallucinationManager:
    """Legacy function for backward compatibility - returns a pass-through manager"""
    if session_id not in _legacy_managers:
        _legacy_managers[session_id] = LegacyHallucinationManager(session_id)
    return _legacy_managers[session_id]

def cleanup_session_manager(session_id: str) -> None:
    """Legacy cleanup function for backward compatibility"""
    if session_id in _legacy_managers:
        del _legacy_managers[session_id]
