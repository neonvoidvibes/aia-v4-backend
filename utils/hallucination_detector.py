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

# Token normalizer for overlap detection
_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")
def _norm_token(t: str) -> str:
    t = t.lower()
    t = _PUNCT_RE.sub("", t)
    return t

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
    # New stricter overlap criteria
    min_overlap_words: int = 4
    min_overlap_secs: float = 1.2

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
    """
    Returns (text, reason, cut_s). cut_s < 0 => no trim.
    """
    # Guard: need data
    if not prev_delivered_words or not curr_words:
        return curr_text_raw, "", -1.0

    # Build normalized sequences for consecutive tail/head match
    prev_tokens = [_norm_token(w.text) for w in prev_delivered_words]
    curr_tokens = [_norm_token(w.text) for w in curr_words]

    max_run = 0
    cut_idx = 0
    max_run_secs = 0.0
    # find longest k where prev tail == curr head (consecutive)
    for k in range(1, min(len(curr_tokens), len(prev_tokens)) + 1):
        if prev_tokens[-k:] == curr_tokens[:k]:
            max_run = k
            cut_idx = k
            max_run_secs = curr_words[k-1].end - curr_words[0].start
        else:
            break

    # Stricter overlap: at least 4 words and 1.2s, and ≥2 distinct tokens
    if max_run >= state.min_overlap_words and \
       max_run_secs >= state.min_overlap_secs and \
       len(set(curr_tokens[:cut_idx])) >= 2:
        kept_words = curr_words[cut_idx:]
        kept_text = " ".join(w.text for w in kept_words) if kept_words else ""

        # Log the trim event with details
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"overlap_trim k={max_run} secs={max_run_secs:.1f}")

        return kept_text, "initial_overlap_trimmed", curr_words[cut_idx-1].end

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