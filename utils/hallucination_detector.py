"""
Session-aware initial-overlap filter for streaming ASR.
Trims only when the head of the current segment overlaps the tail of the previous
in content words, with confidence + temporal gates, and a cooldown to prevent cascades.
"""

from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter
import re

STOPWORDS = set("""
a an the and or but if then so of in on at to for with from by as is are was were be been being it's its
""".split())
NUMERIC_RE = re.compile(r"\d")
PUNCT_RE = re.compile(r"[^\w']+")

@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds
    conf: float   # 0..1

@dataclass
class DetectorState:
    prev_tail: List[Word]
    cooldown_until_ts: float
    # Tunables (reasonable defaults)
    tail_window_s: float = 1.8
    head_window_s: float = 1.8
    min_tokens: int = 3
    jaccard_thresh: float = 0.55
    prefix_char_sim_thresh: float = 0.80
    min_conf: float = 0.35
    cooldown_s: float = 6.0

def _content_tokens(s: str) -> List[str]:
    toks = [t.lower() for t in PUNCT_RE.sub(" ", s).split() if t]
    return [t for t in toks if t not in STOPWORDS]

def _window(words: List[Word], start: float, end: float) -> List[Word]:
    return [w for w in words if w.start < end and w.end > start]

def _slice_tail(words: List[Word], window_s: float) -> List[Word]:
    if not words: return []
    end = max(w.end for w in words)
    start = end - window_s
    return _window(words, start, end)

def _slice_head(words: List[Word], window_s: float) -> List[Word]:
    if not words: return []
    start = min(w.start for w in words)
    end = start + window_s
    return _window(words, start, end)

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = Counter(a), Counter(b)
    inter = sum((sa & sb).values())
    union = sum((sa | sb).values())
    return inter / union if union else 0.0

def _prefix_char_sim(a: str, b: str) -> float:
    A, B = a[:30], b[:30]
    dp = [[0]*(len(B)+1) for _ in range(len(A)+1)]
    for i in range(len(A)):
        for j in range(len(B)):
            dp[i+1][j+1] = dp[i][j]+1 if A[i]==B[j] else max(dp[i][j+1], dp[i+1][j])
    lcs = dp[-1][-1]
    return lcs / max(1, min(len(A), len(B)))

def maybe_trim_repetition(
    state: DetectorState,
    stream_time_s: float,
    prev_delivered_words: List[Word],
    curr_words: List[Word],
    curr_text_raw: str
) -> Tuple[str, str, float]:
    """Return (text, reason, trimmed_until_s). Empty reason => no change; trimmed_until_s=-1."""
    if stream_time_s < state.cooldown_until_ts:
        return curr_text_raw, "", -1.0

    tail = _slice_tail(prev_delivered_words, state.tail_window_s)
    head = _slice_head(curr_words, state.head_window_s)
    if not tail or not head:
        return curr_text_raw, "", -1.0

    tail_text = " ".join(w.text for w in tail if w.conf >= state.min_conf)
    head_text = " ".join(w.text for w in head if w.conf >= state.min_conf)
    if NUMERIC_RE.search(head_text):
        return curr_text_raw, "", -1.0

    tail_tokens = _content_tokens(tail_text)
    head_tokens = _content_tokens(head_text)
    if len(head_tokens) < state.min_tokens:
        return curr_text_raw, "", -1.0

    jacc = _jaccard(tail_tokens, head_tokens)
    pfx = _prefix_char_sim(tail_text.lower(), head_text.lower())
    if jacc < state.jaccard_thresh or pfx < state.prefix_char_sim_thresh:
        return curr_text_raw, "", -1.0

    # Trim at most the first ~head_window_s seconds of the current segment
    head_last_end = max(w.end for w in head)
    kept_words = [w for w in curr_words if w.end > head_last_end]
    kept_text = " ".join(w.text for w in kept_words) if kept_words else ""

    state.cooldown_until_ts = stream_time_s + state.cooldown_s
    if kept_text and len(kept_text) >= 0.5 * max(1, len(curr_text_raw)):
        return kept_text, "initial_overlap_trimmed", head_last_end
    return curr_text_raw, "", -1.0

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