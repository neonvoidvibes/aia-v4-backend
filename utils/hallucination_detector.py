"""
Strict subtractive overlap filter for streaming ASR.
NEVER prepends/carries seed words - only removes overlap.
Includes seed-head repeat dampener to prevent chant loops.
"""

from dataclasses import dataclass
from typing import List, Tuple, Deque
from collections import deque
import re

# Configuration constants
_MIN_OVERLAP_TOKENS = 4      # tighten overlap to avoid false matches
_MAX_CTX_TOKENS = 15         # compare last/first N tokens around boundary
_HEAD_DEDUP_N = 3            # up to 3 tokens considered a "head"
_HEAD_CACHE_K = 6            # remember recent heads to dampen repeats
_HEAD_DEDUP_MIN_TOKENS = 12  # guard: don't dedup very short segments
_SHORT_SEGMENT_DURATION_S = 8.0
_SHORT_SEGMENT_CTX_FACTOR = 0.5

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

@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds
    conf: float   # 0..1

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
        self.recent_heads = deque(maxlen=_HEAD_CACHE_K)

def maybe_trim_repetition(
    state: DetectorState,
    stream_time_s: float,
    prev_delivered_words: List[Word],
    curr_words: List[Word],
    curr_text_raw: str
) -> Tuple[str, str, float]:
    """
    Strict subtractive stitcher - ONLY removes overlap, never prepends.
    Includes seed-head repeat dampener to prevent chant loops.
    Returns (text, reason, cut_s). cut_s < 0 => no trim.
    """
    import logging
    logger = logging.getLogger(__name__)

    new_tokens = _normalize_tokens(curr_text_raw)
    if not new_tokens:
        return curr_text_raw, "", -1.0

    # 1) Trim overlap strictly (subtractive only)
    max_ctx_tokens = _MAX_CTX_TOKENS
    min_overlap_tokens = _MIN_OVERLAP_TOKENS
    if curr_words:
        segment_duration = max(curr_words[-1].end - curr_words[0].start, 0.0)
        if segment_duration <= _SHORT_SEGMENT_DURATION_S:
            max_ctx_tokens = max(2, int(_MAX_CTX_TOKENS * _SHORT_SEGMENT_CTX_FACTOR))
            min_overlap_tokens = max(2, min(_MIN_OVERLAP_TOKENS, max_ctx_tokens))

    tail = state.last_tokens[-max_ctx_tokens:]
    head = new_tokens[:max_ctx_tokens]
    cut_idx = 0

    if tail and head:
        l = _lcs_len(tail, head)
        if l >= min_overlap_tokens:
            # Find the longest matching contiguous prefix
            k = min(len(head), len(tail))
            for i in range(k, 0, -1):
                if head[:i] == tail[-i:]:
                    cut_idx = i
                    break

            if cut_idx >= min_overlap_tokens:
                new_tokens = new_tokens[cut_idx:]
                logger.info(f"overlap_trim removed {cut_idx} tokens from head")

    # 2) Seed-head repeat dampener (don't allow repeated heads)
    original_tokens = new_tokens[:]
    dedup_guard_reason = ""
    if new_tokens:
        head_tuple = tuple(new_tokens[:_HEAD_DEDUP_N])
        token_count = len(new_tokens)
        should_guard = (
            token_count <= _HEAD_DEDUP_N or
            token_count < _HEAD_DEDUP_MIN_TOKENS or
            (_HEAD_DEDUP_N / max(token_count, 1)) > 0.5
        )

        if head_tuple in state.recent_heads and not should_guard:
            # drop exact repeated head once
            new_tokens = new_tokens[_HEAD_DEDUP_N:]
            logger.info(f"head_dedup removed repeated {_HEAD_DEDUP_N} tokens: {head_tuple}")
        elif head_tuple in state.recent_heads and should_guard:
            dedup_guard_reason = "head_guard"
            logger.debug(
                "head_dedup guard prevented removal: tokens=%s", token_count
            )

        # Update head cache AFTER potential drop (or guard)
        if new_tokens:
            state.recent_heads.append(tuple(new_tokens[:_HEAD_DEDUP_N]))

    restored_original = False
    if not new_tokens and original_tokens:
        logger.info("segment trimming produced empty tokens; restoring original text")
        new_tokens = original_tokens
        cut_idx = 0
        dedup_guard_reason = dedup_guard_reason or "restored_original"
        restored_original = True

    # Calculate cut boundary for word-level trimming
    cut_s = -1.0
    if not restored_original and len(original_tokens) > len(new_tokens) and curr_words:
        # Find the word boundary where we cut
        words_cut = len(original_tokens) - len(new_tokens)
        if words_cut < len(curr_words):
            cut_s = curr_words[words_cut - 1].end if words_cut > 0 else curr_words[0].start

    final_text = " ".join(new_tokens)

    # Update rolling context
    state.last_tokens = (state.last_tokens + new_tokens)[-200:]

    reason = ""
    if restored_original:
        reason = dedup_guard_reason or "restored_original"
    elif cut_idx > 0 or len(original_tokens) > len(new_tokens):
        reason = "overlap_trimmed"
    elif dedup_guard_reason:
        reason = dedup_guard_reason
    return final_text, reason, cut_s

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
