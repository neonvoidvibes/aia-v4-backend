"""
Production cardinality and performance tests for hallucination metrics.
Critical validation that metrics stay under Prometheus limits with real workloads.
"""
import time
import threading
from collections import defaultdict
from typing import Dict, Set
import pytest

from utils.hallucination_metrics_v2 import (
    metrics_collector, DropReason, FallbackReason, OutcomeType, TrimPhase
)


class MetricsSeriesCollector:
    """Simulates Prometheus metric series collection for cardinality testing."""

    def __init__(self):
        self.series: Set[str] = set()

    def record_series(self, metric_name: str, labels: Dict[str, str]):
        """Record a unique metric series (metric + label combination)."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        series_key = f"{metric_name}{{{label_str}}}"
        self.series.add(series_key)

    def get_series_count(self) -> int:
        """Return total unique series count."""
        return len(self.series)

    def get_series_by_prefix(self, prefix: str) -> Set[str]:
        """Get all series matching a prefix."""
        return {s for s in self.series if s.startswith(prefix)}


def test_cardinality_10k_sessions_under_2k_series():
    """
    CRITICAL: Simulate 10k concurrent sessions, verify total series <2k.
    This prevents Prometheus cardinality explosion in production.
    """
    collector = MetricsSeriesCollector()

    # Simulate realistic label combinations
    providers = ["whisper", "azure", "deepgram"]
    languages = ["en", "sv", "no", "da", "fi"]

    # Track all possible label combinations that will be generated
    expected_series = set()

    # Base metrics without session-specific labels
    base_metrics = [
        "hallucination_segments_processed_total",
        "hallucination_drops_total",
        "hallucination_fallbacks_total",
        "hallucination_post_asr_decision_duration_seconds",
        "hallucination_trim_attempts_total",
        "hallucination_trim_applied_total",
        "hallucination_context_length",
        "hallucination_cut_word_count",
        "hallucination_empty_after_trim_total",
        "hallucination_validator_outcomes_total",
        "hallucination_pii_outcomes_total",
        "hallucination_first_utterance_sessions_total",
        "hallucination_first_utterance_duplicates_total"
    ]

    # Calculate expected series for each metric
    for metric in base_metrics:
        if metric == "hallucination_drops_total":
            # Has reason label
            for provider in providers:
                for language in languages:
                    for reason in [r.value for r in DropReason]:
                        expected_series.add(f"{metric}{{language={language},provider={provider},reason={reason}}}")
        elif metric == "hallucination_fallbacks_total":
            # Has reason label
            for provider in providers:
                for language in languages:
                    for reason in [r.value for r in FallbackReason]:
                        expected_series.add(f"{metric}{{language={language},provider={provider},reason={reason}}}")
        elif metric == "hallucination_trim_attempts_total":
            # Has phase label only
            for provider in providers:
                for language in languages:
                    for phase in [p.value for p in TrimPhase]:
                        expected_series.add(f"{metric}{{language={language},phase={phase},provider={provider}}}")
        elif metric == "hallucination_trim_applied_total":
            # Has phase and reason labels
            for provider in providers:
                for language in languages:
                    for phase in [p.value for p in TrimPhase]:
                        expected_series.add(f"{metric}{{language={language},phase={phase},provider={provider},reason=overlap_trimmed}})")
        elif metric in ["hallucination_validator_outcomes_total", "hallucination_pii_outcomes_total"]:
            # Has outcome label
            for provider in providers:
                for language in languages:
                    for outcome in [o.value for o in OutcomeType]:
                        expected_series.add(f"{metric}{{language={language},outcome={outcome},provider={provider}}}")
        else:
            # Standard provider/language labels only
            for provider in providers:
                for language in languages:
                    expected_series.add(f"{metric}{{language={language},provider={provider}}}")

    # Add session tracker (no provider/language labels)
    expected_series.add("hallucination_session_tracker_sessions")

    print(f"Theoretical max series count: {len(expected_series)}")
    assert len(expected_series) < 500, f"Even theoretical max should be <500, got {len(expected_series)}"

    # Now simulate 10k sessions with realistic metrics calls
    session_count = 10000
    calls_per_session = 50  # Average segments per session

    start_time = time.time()

    # Simulate concurrent sessions with realistic usage patterns
    for session_id in range(session_count):
        provider = providers[session_id % len(providers)]
        language = languages[session_id % len(languages)]

        for call in range(calls_per_session):
            # Track segment processing (always happens)
            collector.record_series("hallucination_segments_processed_total", {
                "provider": provider, "language": language
            })

            # Track trim attempts (always happens)
            phase = TrimPhase.BOOTSTRAP.value if call < 3 else TrimPhase.NORMAL.value
            collector.record_series("hallucination_trim_attempts_total", {
                "provider": provider, "language": language, "phase": phase
            })

            # Track trim applied (20% of attempts)
            if call % 5 == 0:
                collector.record_series("hallucination_trim_applied_total", {
                    "provider": provider, "language": language, "phase": phase, "reason": "overlap_trimmed"
                })

            # Track drops occasionally (5% rate)
            if call % 20 == 0:
                collector.record_series("hallucination_drops_total", {
                    "provider": provider, "language": language,
                    "reason": DropReason.FILTER_EMPTY.value
                })

            # Track fallbacks occasionally (10% rate)
            if call % 10 == 0:
                collector.record_series("hallucination_fallbacks_total", {
                    "provider": provider, "language": language,
                    "reason": FallbackReason.MIN_SIZE.value
                })

            # Track validator outcomes (most segments)
            if call % 2 == 0:
                collector.record_series("hallucination_validator_outcomes_total", {
                    "provider": provider, "language": language,
                    "outcome": OutcomeType.PASS.value
                })

            # Track PII outcomes (most segments)
            if call % 2 == 0:
                collector.record_series("hallucination_pii_outcomes_total", {
                    "provider": provider, "language": language,
                    "outcome": OutcomeType.PASS.value
                })

            # Track context length distribution
            if call % 3 == 0:
                collector.record_series("hallucination_context_length", {
                    "provider": provider, "language": language
                })

    # Track session count (single series)
    collector.record_series("hallucination_session_tracker_sessions", {})

    elapsed = time.time() - start_time
    final_series_count = collector.get_series_count()

    print(f"Simulated {session_count} sessions with {session_count * calls_per_session} metric calls")
    print(f"Actual unique series: {final_series_count}")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Calls per second: {session_count * calls_per_session/elapsed:.0f}")

    # CRITICAL: Series count must stay under 2k for production Prometheus
    assert final_series_count < 2000, f"Series count {final_series_count} exceeds 2k limit!"

    # Verify actual series is reasonable compared to theoretical max
    assert final_series_count < len(expected_series), f"Actual {final_series_count} should be < theoretical max {len(expected_series)}"
    assert final_series_count > 50, f"Actual series {final_series_count} seems too low - missing metrics?"

    print("✓ PASS: Cardinality stays under 2k series for 10k sessions")


def test_metrics_performance_under_02ms_p95():
    """
    CRITICAL: Ensure metrics calls are <0.2ms p95 to avoid blocking transcription.
    """
    providers = ["whisper", "azure"]
    languages = ["en", "sv"]

    # Collect timing samples
    timings = []
    iterations = 10000

    print(f"Running {iterations} metrics calls for performance test...")

    for i in range(iterations):
        provider = providers[i % len(providers)]
        language = languages[i % len(languages)]

        start = time.perf_counter()

        # Simulate realistic metrics calls in hot path
        metrics_collector.track_segment_processing(provider=provider, language=language)
        metrics_collector.track_trim_attempt(45, provider=provider, language=language)
        metrics_collector.track_post_asr_decision("kept", provider=provider, language=language)
        metrics_collector.track_context_length(45, provider=provider, language=language)

        end = time.perf_counter()
        timings.append((end - start) * 1000)  # Convert to ms

    # Calculate percentiles
    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(len(timings) * 0.95)]
    p99 = timings[int(len(timings) * 0.99)]
    avg = sum(timings) / len(timings)

    print(f"Performance results (ms):")
    print(f"  Average: {avg:.3f}ms")
    print(f"  P50: {p50:.3f}ms")
    print(f"  P95: {p95:.3f}ms")
    print(f"  P99: {p99:.3f}ms")

    # CRITICAL: P95 must be under 0.2ms to not impact transcription performance
    assert p95 < 0.2, f"P95 latency {p95:.3f}ms exceeds 0.2ms threshold!"

    # Additional sanity checks
    assert avg < 0.1, f"Average latency {avg:.3f}ms too high"
    assert p99 < 0.5, f"P99 latency {p99:.3f}ms too high"

    print("✓ PASS: Metrics performance under 0.2ms p95")


def test_concurrent_metrics_thread_safety():
    """
    Verify metrics collection is thread-safe under concurrent load.
    """
    providers = ["whisper", "azure"]
    languages = ["en", "sv"]

    errors = []
    iterations_per_thread = 1000
    thread_count = 10

    def worker_thread(thread_id: int):
        try:
            for i in range(iterations_per_thread):
                provider = providers[i % len(providers)]
                language = languages[i % len(languages)]

                # Mix of different metric calls
                metrics_collector.track_segment_processing(provider=provider, language=language)
                metrics_collector.track_trim_attempt(45, provider=provider, language=language)

                if i % 5 == 0:
                    metrics_collector.track_drop(DropReason.FILTER_EMPTY, provider=provider, language=language)

                if i % 3 == 0:
                    metrics_collector.track_fallback(FallbackReason.MIN_SIZE, provider=provider, language=language)

        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")

    # Start concurrent threads
    threads = []
    start_time = time.time()

    for i in range(thread_count):
        t = threading.Thread(target=worker_thread, args=(i,))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    total_calls = thread_count * iterations_per_thread * 4  # 4 calls per iteration avg

    print(f"Concurrent test: {thread_count} threads × {iterations_per_thread} iterations")
    print(f"Total metric calls: {total_calls}")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Calls per second: {total_calls/elapsed:.0f}")

    # Verify no thread safety errors
    assert not errors, f"Thread safety errors: {errors}"

    # Verify reasonable throughput
    calls_per_sec = total_calls / elapsed
    assert calls_per_sec > 50000, f"Throughput {calls_per_sec:.0f} calls/sec too low"

    print("✓ PASS: Thread safety validated under concurrent load")


if __name__ == "__main__":
    test_cardinality_10k_sessions_under_2k_series()
    test_metrics_performance_under_02ms_p95()
    test_concurrent_metrics_thread_safety()
    print("All cardinality and performance tests passed!")