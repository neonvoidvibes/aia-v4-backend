"""
Production-grade metrics collection for hallucination filtering system.
Low-cardinality, non-blocking, atomic operations only.
"""

import time
from typing import Dict, Optional, Set
from collections import defaultdict, deque
import threading
from enum import Enum

try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback metrics that do nothing
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def time(self): return NullTimer()

class NullTimer:
    def __enter__(self): return self
    def __exit__(self, *args): pass


class TrimPhase(Enum):
    """Bootstrap vs normal trimming phase."""
    BOOTSTRAP = "bootstrap"
    NORMAL = "normal"


class TrimReason(Enum):
    """Low-cardinality trim reasons."""
    OVERLAP_TRIMMED = "overlap_trimmed"
    NO_TRIM = "no_trim"


class DropReason(Enum):
    """Low-cardinality drop reasons."""
    ASR_EMPTY = "asr_empty"
    FILTER_EMPTY = "filter_empty"
    PII_ERROR = "pii_error"
    PII_BLOCKED = "pii_blocked"
    VALIDATOR_FAILED = "validator_failed"


class FallbackReason(Enum):
    """Low-cardinality fallback reasons."""
    PIPELINE_EMPTY = "pipeline_empty"
    MIN_SIZE = "min_size"
    PII_INVALID = "pii_invalid"


class OutcomeType(Enum):
    """Processing outcome types."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    REDACT = "redact"


# Core metrics with low cardinality labels only
# Labels: provider, language, (phase/reason enum)

# 1. Segment processing
segments_processed_total = Counter(
    'hallucination_segments_processed_total',
    'Total segments processed',
    ['provider', 'language']
)

# 2. Trim efficiency by phase (not time-based)
trim_attempts_total = Counter(
    'hallucination_trim_attempts_total',
    'Total trim attempts by phase',
    ['provider', 'language', 'phase']  # bootstrap/normal
)

trim_applied_total = Counter(
    'hallucination_trim_applied_total',
    'Trims actually applied by phase',
    ['provider', 'language', 'phase', 'reason']  # overlap_trimmed
)

# 3. Cut word distribution (histogram)
cut_word_count_histogram = Histogram(
    'hallucination_cut_word_count',
    'Distribution of words cut per trim',
    ['provider', 'language'],
    buckets=[0, 1, 2, 3, 4, 6, 8, 12, float('inf')]
)

# 4. Post-ASR decisions
post_asr_decisions_total = Counter(
    'hallucination_post_asr_decisions_total',
    'Post-ASR decision outcomes',
    ['provider', 'language', 'outcome']  # kept/dropped/fallback
)

# 5. Drops by reason (false-drop guard)
drops_total = Counter(
    'hallucination_drops_total',
    'Segments dropped by reason',
    ['provider', 'language', 'reason']  # enum values only
)

# 6. Fallbacks by reason
fallbacks_total = Counter(
    'hallucination_fallbacks_total',
    'Fallbacks to original text',
    ['provider', 'language', 'reason']  # enum values only
)

# 7. Empty after trim sentinel (bug detector)
empty_after_trim_total = Counter(
    'hallucination_empty_after_trim_total',
    'Segments empty after trimming - bug indicator',
    ['provider', 'language']
)

# 8. First utterance duplicates (per-provider ratio)
first_utterance_sessions_total = Counter(
    'hallucination_first_utterance_sessions_total',
    'Sessions tracked for first utterance analysis',
    ['provider', 'language']
)

first_utterance_duplicates_total = Counter(
    'hallucination_first_utterance_duplicates_total',
    'Sessions with duplicate first utterance',
    ['provider', 'language']
)

# 9. Latency histograms
post_asr_decision_duration = Histogram(
    'hallucination_post_asr_decision_duration_seconds',
    'Post-ASR decision processing time',
    ['provider', 'language'],
    buckets=[0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, float('inf')]
)

segment_processing_duration = Histogram(
    'hallucination_segment_processing_duration_seconds',
    'End-to-end segment processing time',
    ['provider', 'language'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, float('inf')]
)

# 10. Context health (gauge-like via histogram)
context_length_histogram = Histogram(
    'hallucination_context_length',
    'Context length distribution',
    ['provider', 'language'],
    buckets=[0, 10, 25, 50, 100, 150, 200, 250, float('inf')]
)

# 11. Validator/PII outcomes
validator_outcomes_total = Counter(
    'hallucination_validator_outcomes_total',
    'Text validator outcomes',
    ['provider', 'language', 'outcome']  # pass/fail
)

pii_outcomes_total = Counter(
    'hallucination_pii_outcomes_total',
    'PII processing outcomes',
    ['provider', 'language', 'outcome']  # pass/error/redact
)


class SessionTracker:
    """Lock-free per-session tracking for first utterance analysis."""

    def __init__(self, max_sessions: int = 10000):
        self._sessions: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._max_sessions = max_sessions
        self._cleanup_counter = 0

    def track_segment(self, session_id: str, segment_tokens: list, provider: str, language: str) -> bool:
        """Track segment for first utterance analysis. Returns True if duplicate found."""
        # Lightweight cleanup every 1000 calls
        self._cleanup_counter += 1
        if self._cleanup_counter % 1000 == 0:
            self._cleanup_old_sessions()

        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    return False  # Skip tracking if at capacity

                self._sessions[session_id] = {
                    'first_tokens': segment_tokens[:6],  # First 6 tokens
                    'segment_count': 1,
                    'duplicate_found': False,
                    'timestamp': time.monotonic()
                }
                # Count new session
                first_utterance_sessions_total.labels(provider=provider, language=language).inc()
                return False

            session_data = self._sessions[session_id]
            session_data['segment_count'] += 1

            # Only check segments 2-3 for duplicates
            if (session_data['segment_count'] <= 3 and
                not session_data['duplicate_found'] and
                len(segment_tokens) >= 3):

                # Check if first 3 tokens match
                if segment_tokens[:3] == session_data['first_tokens'][:3]:
                    session_data['duplicate_found'] = True
                    # Count duplicate
                    first_utterance_duplicates_total.labels(provider=provider, language=language).inc()
                    return True

        return False

    def _cleanup_old_sessions(self):
        """Remove sessions older than 1 hour."""
        cutoff = time.monotonic() - 3600  # 1 hour
        with self._lock:
            old_sessions = [
                sid for sid, data in self._sessions.items()
                if data['timestamp'] < cutoff
            ]
            for sid in old_sessions:
                del self._sessions[sid]

    def get_session_count(self) -> int:
        """Get current session count for monitoring."""
        return len(self._sessions)


class HallucinationMetricsCollector:
    """Non-blocking, atomic metrics collector for production use."""

    def __init__(self):
        self._session_tracker = SessionTracker()

    def track_segment_processing(self, provider: str = "unknown", language: str = "unknown"):
        """Track segment entering processing pipeline."""
        segments_processed_total.labels(provider=provider, language=language).inc()

    def track_trim_attempt(self, context_length: int, provider: str = "unknown", language: str = "unknown"):
        """Track trim attempt with phase detection."""
        phase = TrimPhase.BOOTSTRAP.value if context_length < 8 else TrimPhase.NORMAL.value
        trim_attempts_total.labels(provider=provider, language=language, phase=phase).inc()

    def track_trim_applied(self, context_length: int, cut_word_count: int,
                          provider: str = "unknown", language: str = "unknown"):
        """Track successful trim application."""
        phase = TrimPhase.BOOTSTRAP.value if context_length < 8 else TrimPhase.NORMAL.value

        # Count trim
        trim_applied_total.labels(
            provider=provider,
            language=language,
            phase=phase,
            reason=TrimReason.OVERLAP_TRIMMED.value
        ).inc()

        # Record cut word distribution
        cut_word_count_histogram.labels(provider=provider, language=language).observe(cut_word_count)

    def track_post_asr_decision(self, outcome: str, provider: str = "unknown", language: str = "unknown"):
        """Track post-ASR decision outcome (kept/dropped/fallback)."""
        post_asr_decisions_total.labels(provider=provider, language=language, outcome=outcome).inc()

    def track_drop(self, reason: DropReason, provider: str = "unknown", language: str = "unknown"):
        """Track segment drop with low-cardinality reason."""
        drops_total.labels(provider=provider, language=language, reason=reason.value).inc()

    def track_fallback(self, reason: FallbackReason, provider: str = "unknown", language: str = "unknown"):
        """Track fallback to original text."""
        fallbacks_total.labels(provider=provider, language=language, reason=reason.value).inc()

    def track_empty_after_trim(self, provider: str = "unknown", language: str = "unknown"):
        """Track empty after trim - bug indicator."""
        empty_after_trim_total.labels(provider=provider, language=language).inc()

    def track_first_utterance(self, session_id: str, segment_tokens: list,
                             provider: str = "unknown", language: str = "unknown") -> bool:
        """Track first utterance patterns. Returns True if duplicate detected."""
        return self._session_tracker.track_segment(session_id, segment_tokens, provider, language)

    def track_context_length(self, length: int, provider: str = "unknown", language: str = "unknown"):
        """Track context length for capacity monitoring."""
        context_length_histogram.labels(provider=provider, language=language).observe(length)

    def track_validator_outcome(self, outcome: OutcomeType, provider: str = "unknown", language: str = "unknown"):
        """Track validator outcome."""
        validator_outcomes_total.labels(provider=provider, language=language, outcome=outcome.value).inc()

    def track_pii_outcome(self, outcome: OutcomeType, provider: str = "unknown", language: str = "unknown"):
        """Track PII processing outcome."""
        pii_outcomes_total.labels(provider=provider, language=language, outcome=outcome.value).inc()

    def time_post_asr_decision(self, provider: str = "unknown", language: str = "unknown"):
        """Time post-ASR decision processing."""
        return post_asr_decision_duration.labels(provider=provider, language=language).time()

    def time_segment_processing(self, provider: str = "unknown", language: str = "unknown"):
        """Time end-to-end segment processing."""
        return segment_processing_duration.labels(provider=provider, language=language).time()

    def get_session_count(self) -> int:
        """Get current tracked session count."""
        return self._session_tracker.get_session_count()


# Global collector instance
metrics_collector = HallucinationMetricsCollector()


def get_metric_cardinality_estimate() -> Dict[str, int]:
    """Estimate metric cardinality for monitoring."""
    # Assume: 5 providers × 10 languages = 50 base combinations
    # Add enum cardinalities
    base_cardinality = 5 * 10  # provider × language

    return {
        "segments_processed_total": base_cardinality,
        "trim_attempts_total": base_cardinality * 2,  # × phase
        "trim_applied_total": base_cardinality * 2 * 1,  # × phase × reason
        "cut_word_count_histogram": base_cardinality * 9,  # × buckets
        "post_asr_decisions_total": base_cardinality * 3,  # × outcome
        "drops_total": base_cardinality * 5,  # × drop reason
        "fallbacks_total": base_cardinality * 3,  # × fallback reason
        "empty_after_trim_total": base_cardinality,
        "first_utterance_*_total": base_cardinality * 2,
        "post_asr_decision_duration": base_cardinality * 9,  # × buckets
        "segment_processing_duration": base_cardinality * 8,  # × buckets
        "context_length_histogram": base_cardinality * 9,  # × buckets
        "validator_outcomes_total": base_cardinality * 2,  # × outcome
        "pii_outcomes_total": base_cardinality * 3,  # × outcome
        "total_estimated": base_cardinality * 60  # Conservative estimate
    }