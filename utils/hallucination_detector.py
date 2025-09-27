"""
Strict subtractive overlap filter for streaming ASR.
NEVER prepends/carries seed words - only removes overlap.
Includes seed-head repeat dampener to prevent chant loops.
"""

from dataclasses import dataclass
from typing import List, Tuple, Deque, Optional
from collections import deque
import re
import time
import logging
import threading

# Setup logger
logger = logging.getLogger(__name__)

# Feature flag for debug logging (can be set via environment or config)
ENABLE_DETECTOR_DEBUG = False
try:
    from utils.feature_flags import feature_enabled
    ENABLE_DETECTOR_DEBUG = feature_enabled('hallucination.detector_debug', False)
except ImportError:
    pass

try:
    from .hallucination_metrics_v2 import metrics_collector
except ImportError:
    # Fallback if metrics not available
    class NullMetricsCollector:
        def track_trim_attempt(self, *args, **kwargs): pass
        def track_trim_applied(self, *args, **kwargs): pass
        def track_empty_after_trim(self, *args, **kwargs): pass
        def track_context_length(self, *args, **kwargs): pass
    metrics_collector = NullMetricsCollector()

# Configuration constants
# Lower minimum so early repeats like "hallå nu testar jag" are trimmed.
_MIN_OVERLAP_TOKENS = 3
_MAX_CONTEXT_TOKENS = 50
_MIN_SEGMENT_TOKENS = 6
# Keep ratio; bootstrap will relax it briefly.
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
    """Compute overlap between context and current segment head using sliding window."""
    if not tail or not head:
        return 0

    # First try traditional suffix-prefix overlap (most common case)
    max_overlap = 0
    k = min(len(tail), len(head))
    for i in range(1, k + 1):
        if tail[-i:] == head[:i]:
            max_overlap = i

    # If no suffix-prefix overlap, check for internal repetitions using sliding window
    if max_overlap == 0:
        # Look for head pattern anywhere in the tail context
        head_len = len(head)
        tail_len = len(tail)

        # Check all possible positions in tail for head pattern
        for start_pos in range(tail_len - head_len + 1):
            # Check for match at this position
            match_len = 0
            for i in range(min(head_len, tail_len - start_pos)):
                if tail[start_pos + i] == head[i]:
                    match_len += 1
                else:
                    break

            # Update max overlap if we found a longer match
            if match_len >= 3:  # Minimum meaningful overlap
                max_overlap = max(max_overlap, match_len)

    if ENABLE_DETECTOR_DEBUG:
        logger.debug(f"_lcs_overlap result={max_overlap} (suffix-prefix or sliding window)")
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
    Never prepends/carries seed words. Includes repeat dampener with n-gram tracking.
    Thread-safe with light locking for concurrent segment processing.
    """
    last_tokens: List[str]
    recent_heads: Deque[Tuple[str, ...]]
    recent_ngrams: Deque[Tuple[Tuple[str, ...], float]]  # (ngram, timestamp)
    _lock: threading.Lock

    def __init__(self):
        self.last_tokens = []
        self.recent_heads = deque(maxlen=10)
        self.recent_ngrams = deque(maxlen=50)  # Track more n-grams with timestamps
        self._lock = threading.Lock()

    def tail_tokens(self, limit: int) -> List[str]:
        """Return last N tokens from context."""
        with self._lock:
            return self.last_tokens[-limit:] if self.last_tokens else []

    def add_ngram_pattern(self, ngram: Tuple[str, ...], timestamp: float = None) -> None:
        """Add n-gram pattern to recent tracking with timestamp."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self.recent_ngrams.append((ngram, timestamp))

    def check_ngram_loops(self, tokens: List[str], lookback_seconds: float = 10.0) -> bool:
        """Check if current tokens contain repeating n-gram patterns within time window."""
        if len(tokens) < 2:
            return False

        current_time = time.time()
        # Generate n-grams from current tokens (2-4 grams)
        for n in range(2, min(5, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i+n])

                # Check against recent n-grams within time window
                with self._lock:
                    for recent_gram, timestamp in self.recent_ngrams:
                        if current_time - timestamp <= lookback_seconds:
                            if recent_gram == gram:
                                return True

        return False

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

    # DEBUG: Log overlap detection details
    if ENABLE_DETECTOR_DEBUG:
        logger.debug(f"ctx_tail={ctx_tail[-10:] if ctx_tail else []}, seg_head={seg_head[:10]}")
        logger.debug(f"context_len={len(state.last_tokens)}, seg_head_len={len(seg_head)}")

    # Check for n-gram loops before processing overlap
    if seg_head and state.check_ngram_loops(seg_head):
        if ENABLE_DETECTOR_DEBUG:
            logger.debug(f"detected repeating n-gram pattern, trimming entire segment")
        # Trim the entire segment as it's a detected loop
        with state._lock:
            state.last_tokens = (state.last_tokens + seg_head)[-200:]  # Still update context
        return [], "ngram_loop_detected", len(words)

    if not ctx_tail or not seg_head:
        # Still update context even if no overlap detection
        head_norm = [w.norm for w in words if w.norm]
        with state._lock:
            state.last_tokens = (state.last_tokens + head_norm)[-200:]
        # Track n-grams from processed tokens
        _track_ngrams_from_tokens(state, head_norm)
        if ENABLE_DETECTOR_DEBUG:
            logger.debug(f"No ctx_tail or seg_head, updated context to len={len(state.last_tokens)}")
        return words, None, 0

    # Track trim attempt with phase detection
    context_len = len(state.last_tokens)
    # Note: metrics_collector handles provider/language from call site

    # Compute suffix-prefix overlap length (context suffix vs current head).
    overlap = _lcs_suffix_prefix_overlap(ctx_tail, seg_head)
    if ENABLE_DETECTOR_DEBUG:
        logger.debug(f"suffix-prefix overlap={overlap} between ctx_tail={ctx_tail} and seg_head={seg_head}")

    # Bootstrap: in the first few tokens of context, use a relaxed threshold so
    # immediate repeats of the greeting are trimmed cleanly.
    # This avoids needing time-based logic and keeps trimming purely boundary-based.
    if context_len < 8:
        boot_min = 2
        boot_ratio = 0.50
        threshold = max(boot_min, int(len(seg_head) * boot_ratio))
        if ENABLE_DETECTOR_DEBUG:
            logger.debug(f"BOOTSTRAP mode - threshold={threshold} (boot_min={boot_min}, seg_head_len*{boot_ratio}={int(len(seg_head) * boot_ratio)})")
    else:
        threshold = max(min_overlap_tokens, int(len(seg_head) * overlap_ratio))
        if ENABLE_DETECTOR_DEBUG:
            logger.debug(f"NORMAL mode - threshold={threshold} (min_tokens={min_overlap_tokens}, seg_head_len*{overlap_ratio}={int(len(seg_head) * overlap_ratio)})")

    if ENABLE_DETECTOR_DEBUG:
        logger.debug(f"overlap={overlap} vs threshold={threshold}, will_trim={overlap >= threshold}")
    if overlap >= threshold:
        cut = min(overlap, len(words))
        trimmed = words[cut:]
        reason = f"trimmed_overlap:{overlap}"

        # Track empty-after-trim sentinel (bug detector)
        if not trimmed:
            # For streaming mode: prefer empty result over restoring original repetitive text
            # This prevents "chant resurrection" - return empty with reason instead
            if ENABLE_DETECTOR_DEBUG:
                logger.debug(f"full trim resulted in empty, blocking original restore with reason=restored_original_blocked")
            # Note: provider/language passed from call site via thread-local or params
            pass  # Will be tracked at call site

        # update rolling context with the tokens we will KEEP
        kept_norm = [w.norm for w in trimmed if w.norm]
        with state._lock:
            state.last_tokens = (state.last_tokens + kept_norm)[-200:]

        # Track n-grams from kept tokens
        _track_ngrams_from_tokens(state, kept_norm)

        return trimmed, reason, cut

    # no trim; still advance context with head (bounded)
    head_norm = [w.norm for w in words if w.norm]
    with state._lock:
        state.last_tokens = (state.last_tokens + head_norm)[-200:]

    # Track n-grams from all tokens
    _track_ngrams_from_tokens(state, head_norm)

    return words, None, 0

def _track_ngrams_from_tokens(state: DetectorState, tokens: List[str]) -> None:
    """Helper function to track n-grams from processed tokens."""
    if len(tokens) < 2:
        return

    current_time = time.time()
    # Track 2-4 grams for loop detection
    for n in range(2, min(5, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i+n])
            state.add_ngram_pattern(gram, current_time)

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

def _compute_ngram_jaccard(tokens1: List[str], tokens2: List[str], n: int) -> float:
    """Compute Jaccard similarity between n-gram sets of two token sequences."""
    if len(tokens1) < n or len(tokens2) < n:
        return 0.0

    # Generate n-grams
    ngrams1 = set()
    ngrams2 = set()

    for i in range(len(tokens1) - n + 1):
        ngrams1.add(tuple(tokens1[i:i+n]))

    for i in range(len(tokens2) - n + 1):
        ngrams2.add(tuple(tokens2[i:i+n]))

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0

def detect_repetition_in_text(text: str, state: DetectorState) -> bool:
    """
    Advanced repetition check for fallback logic using n-gram similarity.
    Returns True if text contains repetitive patterns that should prevent fallback to original.
    """
    if not text or not text.strip():
        return False

    # Convert text to normalized tokens
    tokens = _normalize_tokens(text)
    if len(tokens) < 2:  # Need at least 2 tokens for repetition
        return False

    # Special case: check for short bigram chants like "hi hi" before performance guard
    if len(tokens) < 6:
        # Check for immediate bigram repetition in short sequences
        for i in range(len(tokens) - 1):
            bigram = tuple(tokens[i:i+2])
            # Check if this bigram appears again later
            for j in range(i+2, len(tokens)):
                if j+1 < len(tokens) and tuple(tokens[j:j+2]) == bigram:
                    if ENABLE_DETECTOR_DEBUG:
                        logger.debug(f"detected short bigram repetition: {bigram}")
                    return True

        # Also check for direct adjacent token repetition
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                if ENABLE_DETECTOR_DEBUG:
                    logger.debug(f"detected adjacent token repetition: {tokens[i]}")
                return True

        return False  # No repetition found in short sequence

    # Check for immediate repetition patterns using n-gram similarity
    for n in range(2, min(5, len(tokens) + 1)):
        for i in range(len(tokens) - n):
            gram = tuple(tokens[i:i+n])
            # Check if this n-gram appears again in the rest of the text
            for j in range(i+n, len(tokens)-n+1):
                if tuple(tokens[j:j+n]) == gram:
                    return True

    # Enhanced cross-segment repetition using n-gram Jaccard similarity
    if state.last_tokens:
        ctx_tail = state.tail_tokens(limit=30)  # Increased context window
        head_tokens = tokens[:min(15, len(tokens))]  # Increased head window

        if len(ctx_tail) >= 3 and len(head_tokens) >= 3:
            # Check traditional overlap first (faster)
            overlap = _lcs_suffix_prefix_overlap(ctx_tail, head_tokens)
            if overlap >= 3:  # Minimum meaningful repetition - early exit
                return True

            # Only do expensive Jaccard if traditional overlap didn't find it
            # Pre-compute n-gram sets to avoid rebuilding in the loop
            ctx_ngram_sets = {}
            head_ngram_sets = {}
            for n in range(2, min(5, len(head_tokens) + 1, len(ctx_tail) + 1)):
                if len(ctx_tail) >= n:
                    ctx_ngram_sets[n] = set(tuple(ctx_tail[i:i+n]) for i in range(len(ctx_tail) - n + 1))
                if len(head_tokens) >= n:
                    head_ngram_sets[n] = set(tuple(head_tokens[i:i+n]) for i in range(len(head_tokens) - n + 1))

            overlapping_ngrams = 0
            strong_match = False
            for n in range(2, min(5, len(head_tokens) + 1, len(ctx_tail) + 1)):
                # Use cached sets for Jaccard computation
                if n not in ctx_ngram_sets or n not in head_ngram_sets:
                    continue

                ngrams1 = ctx_ngram_sets[n]
                ngrams2 = head_ngram_sets[n]
                intersection = len(ngrams1 & ngrams2)
                union = len(ngrams1 | ngrams2)
                jaccard = intersection / union if union > 0 else 0.0

                # Progressive thresholds: more strict for longer n-grams
                threshold = 0.5 if n == 2 else (0.6 if n == 3 else 0.7)
                strong_threshold = 0.8  # Strong evidence threshold

                if jaccard >= threshold:
                    overlapping_ngrams += 1
                    if ENABLE_DETECTOR_DEBUG:
                        logger.debug(f"n={n} jaccard={jaccard:.3f} >= {threshold}")

                # Allow single strong match for trigrams or higher
                if n >= 3 and jaccard >= strong_threshold:
                    strong_match = True
                    if ENABLE_DETECTOR_DEBUG:
                        logger.debug(f"strong n={n} match jaccard={jaccard:.3f} >= {strong_threshold}")

            # Match if either: 2+ different sizes cross thresholds, or single strong trigram+
            if overlapping_ngrams >= 2 or strong_match:
                if ENABLE_DETECTOR_DEBUG:
                    reason = f"{overlapping_ngrams} matching n-gram sizes" if overlapping_ngrams >= 2 else "single strong match"
                    logger.debug(f"detected cross-segment repetition with {reason}")
                return True

    return False
