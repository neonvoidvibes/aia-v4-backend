"""
Comprehensive metrics collection for hallucination filtering system.
Tracks efficiency, safety, performance, and quality metrics for production monitoring.
"""

import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, Enum
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
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Enum:
        def __init__(self, *args, **kwargs): pass
        def state(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

class NullTimer:
    def __enter__(self): return self
    def __exit__(self, *args): pass

logger = logging.getLogger(__name__)

# Session age buckets for trim efficiency analysis
SESSION_AGE_BUCKETS = ["0-5s", "5-30s", "30s+"]

def get_session_age_bucket(session_duration_s: float) -> str:
    """Classify session age into buckets for trim efficiency tracking."""
    if session_duration_s <= 5.0:
        return "0-5s"
    elif session_duration_s <= 30.0:
        return "5-30s"
    else:
        return "30s+"

# Core metrics for hallucination filtering monitoring

# 1. Trim efficiency by session age
trim_efficiency_total = Counter(
    'hallucination_trim_efficiency_total',
    'Total segments processed by session age bucket',
    ['session_age_bucket', 'provider', 'language']
)

trim_efficiency_trimmed = Counter(
    'hallucination_trim_efficiency_trimmed_total',
    'Segments with overlap trimming by session age bucket',
    ['session_age_bucket', 'provider', 'language', 'reason']
)

# 2. False-drop guard (drops that shouldn't happen)
false_drops_total = Counter(
    'hallucination_false_drops_total',
    'Segments dropped by post-ASR pipeline (excluding ASR_EMPTY)',
    ['drop_reason', 'provider', 'language']
)

segments_processed_total = Counter(
    'hallucination_segments_processed_total',
    'Total segments processed through pipeline',
    ['provider', 'language']
)

# 3. Fallback pressure monitoring
fallback_usage_total = Counter(
    'hallucination_fallback_usage_total',
    'Fallback to original text usage',
    ['fallback_reason', 'provider', 'language']
)

# 4. Empty-after-trim sentinel (should be ~0%)
empty_after_trim_total = Counter(
    'hallucination_empty_after_trim_total',
    'Segments that became empty after trimming (bug indicator)',
    ['provider', 'language']
)

# 5. First-utterance duplicate tracking
first_utterance_sessions_total = Counter(
    'hallucination_first_utterance_sessions_total',
    'Total sessions tracked for first utterance analysis',
    ['provider', 'language']
)

first_utterance_duplicates_total = Counter(
    'hallucination_first_utterance_duplicates_total',
    'Sessions with duplicate first utterance in first 3 segments',
    ['provider', 'language', 'duplicate_position']  # position=2nd_segment, 3rd_segment
)

# 6. Kept-vs-cut distribution
cut_word_count_histogram = Histogram(
    'hallucination_cut_word_count',
    'Distribution of cut word count per segment',
    ['provider', 'language'],
    buckets=[0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, float('inf')]
)

# 7. Latency impact monitoring
post_asr_decision_duration = Histogram(
    'hallucination_post_asr_decision_duration_seconds',
    'Time spent in post-ASR decision pipeline',
    ['provider', 'language'],
    buckets=[0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, float('inf')]
)

segment_processing_duration = Histogram(
    'hallucination_segment_processing_duration_seconds',
    'End-to-end segment processing time',
    ['provider', 'language'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float('inf')]
)

# 8. Context health monitoring
context_tail_length = Histogram(
    'hallucination_context_tail_length',
    'Length of context tail used for overlap detection',
    ['provider', 'language'],
    buckets=[0, 5, 10, 20, 50, 100, 150, 200, float('inf')]
)

context_resets_total = Counter(
    'hallucination_context_resets_total',
    'Context resets (indicates session restart or error)',
    ['provider', 'language', 'reason']
)

# 9. Validator/PII outcome tracking
validator_outcomes_total = Counter(
    'hallucination_validator_outcomes_total',
    'Validator decision outcomes',
    ['outcome', 'provider', 'language']  # outcome=pass,fail
)

pii_outcomes_total = Counter(
    'hallucination_pii_outcomes_total',
    'PII processing outcomes',
    ['outcome', 'provider', 'language']  # outcome=pass,error,redact
)


@dataclass
class SessionMetrics:
    """Per-session metrics tracking for first-utterance analysis."""
    session_id: str
    provider: str = "unknown"
    language: str = "unknown"
    session_start_time: float = field(default_factory=time.time)
    segment_count: int = 0
    first_utterance_tokens: Optional[List[str]] = None
    duplicate_found: bool = False

    def __post_init__(self):
        self.creation_time = time.time()


class HallucinationMetricsCollector:
    """Central collector for hallucination filtering metrics."""

    def __init__(self):
        self._session_metrics: Dict[str, SessionMetrics] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 300  # Clean up old sessions every 5 minutes
        self._last_cleanup = time.time()

    def _cleanup_old_sessions(self):
        """Remove session metrics older than 1 hour to prevent memory leaks."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        cutoff_time = current_time - 3600  # 1 hour
        with self._lock:
            old_sessions = [
                session_id for session_id, metrics in self._session_metrics.items()
                if metrics.creation_time < cutoff_time
            ]
            for session_id in old_sessions:
                del self._session_metrics[session_id]

        if old_sessions:
            logger.debug(f"Cleaned up {len(old_sessions)} old session metrics")
        self._last_cleanup = current_time

    def track_segment_processing(self,
                                session_id: str,
                                provider: str = "unknown",
                                language: str = "unknown",
                                session_duration_s: float = 0.0):
        """Track a segment being processed through the pipeline."""
        self._cleanup_old_sessions()

        # Overall processing
        segments_processed_total.labels(provider=provider, language=language).inc()

        # Trim efficiency by session age
        age_bucket = get_session_age_bucket(session_duration_s)
        trim_efficiency_total.labels(
            session_age_bucket=age_bucket,
            provider=provider,
            language=language
        ).inc()

        # Initialize session metrics if needed
        with self._lock:
            if session_id not in self._session_metrics:
                self._session_metrics[session_id] = SessionMetrics(
                    session_id=session_id,
                    provider=provider,
                    language=language
                )

            session_metrics = self._session_metrics[session_id]
            session_metrics.segment_count += 1

    def track_overlap_trim(self,
                          session_id: str,
                          reason: str,
                          cut_word_count: int,
                          session_duration_s: float = 0.0,
                          provider: str = "unknown",
                          language: str = "unknown"):
        """Track overlap trimming occurrence."""
        age_bucket = get_session_age_bucket(session_duration_s)

        # Trim efficiency
        trim_efficiency_trimmed.labels(
            session_age_bucket=age_bucket,
            provider=provider,
            language=language,
            reason=reason
        ).inc()

        # Cut word distribution
        cut_word_count_histogram.labels(provider=provider, language=language).observe(cut_word_count)

    def track_post_asr_decision(self,
                               decision_result: str,  # "kept", "dropped", "fallback"
                               drop_reason: Optional[str] = None,
                               fallback_reason: Optional[str] = None,
                               provider: str = "unknown",
                               language: str = "unknown"):
        """Track post-ASR decision outcomes."""
        if decision_result == "dropped" and drop_reason and drop_reason != "ASR_EMPTY":
            false_drops_total.labels(
                drop_reason=drop_reason,
                provider=provider,
                language=language
            ).inc()

        if decision_result == "fallback" and fallback_reason:
            fallback_usage_total.labels(
                fallback_reason=fallback_reason,
                provider=provider,
                language=language
            ).inc()

    def track_empty_after_trim(self, provider: str = "unknown", language: str = "unknown"):
        """Track segments that became empty after trimming (bug indicator)."""
        empty_after_trim_total.labels(provider=provider, language=language).inc()

    def track_first_utterance(self,
                             session_id: str,
                             segment_tokens: List[str],
                             provider: str = "unknown",
                             language: str = "unknown"):
        """Track first utterance patterns to detect duplicates."""
        with self._lock:
            if session_id not in self._session_metrics:
                return

            session_metrics = self._session_metrics[session_id]

            # Only track first 3 segments
            if session_metrics.segment_count > 3:
                return

            if session_metrics.segment_count == 1:
                # Store first utterance
                session_metrics.first_utterance_tokens = segment_tokens[:6]  # First 6 tokens
                first_utterance_sessions_total.labels(provider=provider, language=language).inc()

            elif session_metrics.segment_count <= 3 and session_metrics.first_utterance_tokens:
                # Check for duplicate in segments 2-3
                current_tokens = segment_tokens[:6]
                if (current_tokens and
                    len(current_tokens) >= 3 and
                    current_tokens[:3] == session_metrics.first_utterance_tokens[:3] and
                    not session_metrics.duplicate_found):

                    # Found duplicate
                    session_metrics.duplicate_found = True
                    position = "2nd_segment" if session_metrics.segment_count == 2 else "3rd_segment"
                    first_utterance_duplicates_total.labels(
                        provider=provider,
                        language=language,
                        duplicate_position=position
                    ).inc()

    def track_context_health(self,
                           context_length: int,
                           provider: str = "unknown",
                           language: str = "unknown"):
        """Track context tail length for health monitoring."""
        context_tail_length.labels(provider=provider, language=language).observe(context_length)

    def track_context_reset(self,
                          reason: str,
                          provider: str = "unknown",
                          language: str = "unknown"):
        """Track context resets."""
        context_resets_total.labels(provider=provider, language=language, reason=reason).inc()

    def track_validator_outcome(self,
                              outcome: str,  # "pass", "fail"
                              provider: str = "unknown",
                              language: str = "unknown"):
        """Track validator outcomes."""
        validator_outcomes_total.labels(outcome=outcome, provider=provider, language=language).inc()

    def track_pii_outcome(self,
                         outcome: str,  # "pass", "error", "redact"
                         provider: str = "unknown",
                         language: str = "unknown"):
        """Track PII processing outcomes."""
        pii_outcomes_total.labels(outcome=outcome, provider=provider, language=language).inc()

    def time_post_asr_decision(self, provider: str = "unknown", language: str = "unknown"):
        """Context manager for timing post-ASR decisions."""
        return post_asr_decision_duration.labels(provider=provider, language=language).time()

    def time_segment_processing(self, provider: str = "unknown", language: str = "unknown"):
        """Context manager for timing end-to-end segment processing."""
        return segment_processing_duration.labels(provider=provider, language=language).time()


# Global collector instance
metrics_collector = HallucinationMetricsCollector()


def get_metrics_summary() -> Dict[str, Any]:
    """Get current metrics summary for dashboards and alerting."""
    # This would typically query the metrics backend (Prometheus)
    # For now, return a placeholder structure
    return {
        "trim_efficiency": {
            "0-5s_bucket_rate": "pending",
            "5-30s_bucket_rate": "pending",
            "30s+_bucket_rate": "pending"
        },
        "safety": {
            "false_drop_rate": "pending",
            "empty_after_trim_rate": "pending"
        },
        "fallback": {
            "fallback_rate": "pending"
        },
        "performance": {
            "post_asr_p95_ms": "pending",
            "segment_p95_ms": "pending"
        },
        "quality": {
            "first_utterance_duplicate_rate": "pending",
            "cut_word_p50": "pending",
            "cut_word_p95": "pending"
        },
        "context": {
            "context_length_p95": "pending",
            "context_reset_rate": "pending"
        }
    }


# Alert thresholds for monitoring
ALERT_THRESHOLDS = {
    "false_drop_rate_critical": 0.01,  # 1%+ spike
    "false_drop_rate_warning": 0.005,  # 0.5%
    "fallback_rate_investigate": 0.08,  # 8%+ drift
    "fallback_rate_warning": 0.05,  # 5%
    "empty_after_trim_any": 0.001,  # Any spike is concerning
    "first_utterance_duplicate_rate": 0.03,  # 3%
    "post_asr_p95_budget_ms": 3.0,  # 3ms p95
    "context_reset_rate_warning": 0.001,  # ~0%
    "validator_fail_rate": 0.003,  # 0.3%
    "pii_error_rate": 0.001,  # ~0%
}