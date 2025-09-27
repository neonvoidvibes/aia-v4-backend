#!/usr/bin/env python3
"""
Test comprehensive metrics collection for hallucination filtering system.
Verifies all 10 monitoring categories are properly instrumented.
"""

import sys
from pathlib import Path
import time
from unittest.mock import Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from utils.hallucination_metrics import (
    metrics_collector, get_session_age_bucket, ALERT_THRESHOLDS
)
from utils.hallucination_detector import DetectorState, Word, maybe_trim_repetition
from services.post_asr_pipeline import (
    decide_transcript_candidate, FeatureToggles, DROP_NONE, DROP_ASR_EMPTY
)


class TestSessionAgeBuckets:
    """Test session age bucket classification."""

    def test_session_age_bucket_classification(self):
        """Test session age bucket boundaries."""
        assert get_session_age_bucket(0.0) == "0-5s"
        assert get_session_age_bucket(2.5) == "0-5s"
        assert get_session_age_bucket(5.0) == "0-5s"
        assert get_session_age_bucket(5.1) == "5-30s"
        assert get_session_age_bucket(15.0) == "5-30s"
        assert get_session_age_bucket(30.0) == "5-30s"
        assert get_session_age_bucket(30.1) == "30s+"
        assert get_session_age_bucket(300.0) == "30s+"


class TestTrimEfficiencyMetrics:
    """Test trim efficiency metrics by session age bucket."""

    def test_segment_processing_tracking(self):
        """Test basic segment processing is tracked."""
        session_id = "test_session_1"

        # Track multiple segments with different ages
        metrics_collector.track_segment_processing(
            session_id, provider="deepgram", language="en", session_duration_s=2.0
        )
        metrics_collector.track_segment_processing(
            session_id, provider="deepgram", language="en", session_duration_s=15.0
        )
        metrics_collector.track_segment_processing(
            session_id, provider="deepgram", language="en", session_duration_s=60.0
        )

    def test_overlap_trim_tracking(self):
        """Test overlap trimming metrics."""
        session_id = "test_session_2"

        # Track trimming in bootstrap phase
        metrics_collector.track_overlap_trim(
            session_id=session_id,
            reason="trimmed_overlap:4",
            cut_word_count=4,
            session_duration_s=1.0,  # 0-5s bucket
            provider="deepgram",
            language="en"
        )

        # Track trimming in normal phase
        metrics_collector.track_overlap_trim(
            session_id=session_id,
            reason="trimmed_overlap:2",
            cut_word_count=2,
            session_duration_s=45.0,  # 30s+ bucket
            provider="deepgram",
            language="en"
        )


class TestPostASRMetrics:
    """Test post-ASR pipeline metrics collection."""

    def test_false_drop_tracking(self):
        """Test false drop detection and tracking."""
        # Simulate various drop scenarios
        metrics_collector.track_post_asr_decision(
            "dropped", drop_reason="DROP_FILTER_EMPTY",
            provider="deepgram", language="en"
        )

        metrics_collector.track_post_asr_decision(
            "dropped", drop_reason="DROP_PII_ERROR",
            provider="whisper", language="sv"
        )

    def test_fallback_pressure_tracking(self):
        """Test fallback usage tracking."""
        # Track different fallback reasons
        metrics_collector.track_post_asr_decision(
            "fallback", fallback_reason="FALLBACK_PIPELINE_EMPTY",
            provider="deepgram", language="en"
        )

        metrics_collector.track_post_asr_decision(
            "fallback", fallback_reason="FALLBACK_MIN_SIZE",
            provider="deepgram", language="en"
        )

    def test_successful_processing_tracking(self):
        """Test successful processing tracking."""
        metrics_collector.track_post_asr_decision(
            "kept", provider="deepgram", language="en"
        )


class TestValidatorPIIMetrics:
    """Test validator and PII outcome tracking."""

    def test_validator_outcomes(self):
        """Test validator pass/fail tracking."""
        metrics_collector.track_validator_outcome("pass", provider="deepgram", language="en")
        metrics_collector.track_validator_outcome("fail", provider="deepgram", language="en")

    def test_pii_outcomes(self):
        """Test PII processing outcome tracking."""
        metrics_collector.track_pii_outcome("pass", provider="deepgram", language="en")
        metrics_collector.track_pii_outcome("error", provider="deepgram", language="en")
        metrics_collector.track_pii_outcome("redact", provider="deepgram", language="en")


class TestFirstUtteranceDuplicates:
    """Test first utterance duplicate detection."""

    def test_first_utterance_tracking(self):
        """Test first utterance duplicate detection."""
        session_id = "test_utterance_session"

        # Simulate first segment
        collector = metrics_collector
        collector._session_metrics[session_id] = collector.SessionMetrics(
            session_id=session_id, provider="deepgram", language="en"
        )
        collector._session_metrics[session_id].segment_count = 1

        # Track first utterance
        collector.track_first_utterance(
            session_id, ["hallå", "nu", "testar", "jag"],
            provider="deepgram", language="en"
        )

        # Simulate second segment with same utterance (duplicate)
        collector._session_metrics[session_id].segment_count = 2
        collector.track_first_utterance(
            session_id, ["hallå", "nu", "testar", "jag"],
            provider="deepgram", language="en"
        )

        # Check duplicate was detected
        assert collector._session_metrics[session_id].duplicate_found

    def test_no_duplicate_detection(self):
        """Test no false duplicate detection."""
        session_id = "test_no_duplicate_session"

        # Simulate different utterances
        collector = metrics_collector
        collector._session_metrics[session_id] = collector.SessionMetrics(
            session_id=session_id, provider="deepgram", language="en"
        )
        collector._session_metrics[session_id].segment_count = 1

        # Track first utterance
        collector.track_first_utterance(
            session_id, ["hallå", "nu", "testar", "jag"],
            provider="deepgram", language="en"
        )

        # Different second utterance
        collector._session_metrics[session_id].segment_count = 2
        collector.track_first_utterance(
            session_id, ["det", "fungerar", "bra"],
            provider="deepgram", language="en"
        )

        # No duplicate should be detected
        assert not collector._session_metrics[session_id].duplicate_found


class TestContextHealthMetrics:
    """Test context health monitoring."""

    def test_context_health_tracking(self):
        """Test context length tracking."""
        metrics_collector.track_context_health(50, provider="deepgram", language="en")
        metrics_collector.track_context_health(150, provider="deepgram", language="en")

    def test_context_reset_tracking(self):
        """Test context reset tracking."""
        metrics_collector.track_context_reset("session_restart", provider="deepgram", language="en")
        metrics_collector.track_context_reset("error", provider="deepgram", language="en")


class TestEmptyAfterTrimSentinel:
    """Test empty-after-trim bug detection."""

    def test_empty_after_trim_tracking(self):
        """Test empty-after-trim sentinel."""
        metrics_collector.track_empty_after_trim(provider="deepgram", language="en")


class TestLatencyMetrics:
    """Test latency monitoring."""

    def test_post_asr_decision_timing(self):
        """Test post-ASR decision timing."""
        with metrics_collector.time_post_asr_decision(provider="deepgram", language="en"):
            time.sleep(0.001)  # Simulate 1ms processing

    def test_segment_processing_timing(self):
        """Test segment processing timing."""
        with metrics_collector.time_segment_processing(provider="deepgram", language="en"):
            time.sleep(0.005)  # Simulate 5ms processing


class TestIntegratedMetrics:
    """Test integrated metrics collection through actual pipeline."""

    def test_overlap_trimming_with_metrics(self):
        """Test overlap trimming generates correct metrics."""
        state = DetectorState()
        state.last_tokens = ["hello", "world", "this", "is"]

        words = [
            Word("this", 0.0, 1.0, 0.9),
            Word("is", 1.0, 2.0, 0.9),
            Word("new", 2.0, 3.0, 0.9),
            Word("content", 3.0, 4.0, 0.9),
        ]

        # This should trigger overlap trimming and metrics
        result_words, reason, cut_count = maybe_trim_repetition(
            words, state, min_overlap_tokens=1, min_segment_tokens=1
        )

        assert len(result_words) == 2  # "new", "content"
        assert reason == "trimmed_overlap:2"

    def test_post_asr_decision_with_metrics(self):
        """Test post-ASR decision generates correct metrics."""
        def dummy_pii(text): return text
        def dummy_validator(text): return bool(text.strip())

        # Successful case
        decision = decide_transcript_candidate(
            original_text="Hello world",
            filtered_text="Hello world",
            run_pii=dummy_pii,
            validator=dummy_validator,
            toggles=FeatureToggles(),
            provider="deepgram",
            language="en"
        )

        assert decision.final_text == "Hello world"
        assert decision.drop_reason == DROP_NONE

    def test_fallback_scenario_with_metrics(self):
        """Test fallback scenario generates correct metrics."""
        def dummy_pii(text): return text
        def dummy_validator(text): return bool(text.strip())

        # Fallback case (empty filtered text)
        decision = decide_transcript_candidate(
            original_text="Original text",
            filtered_text="",  # Empty filtered
            run_pii=dummy_pii,
            validator=dummy_validator,
            toggles=FeatureToggles(never_empty_contract=True),
            provider="deepgram",
            language="en"
        )

        assert decision.final_text == "Original text"
        assert decision.used_fallback


class TestMetricsSummary:
    """Test metrics summary and alerting thresholds."""

    def test_alert_thresholds_defined(self):
        """Test all required alert thresholds are defined."""
        required_thresholds = [
            "false_drop_rate_critical",
            "false_drop_rate_warning",
            "fallback_rate_investigate",
            "fallback_rate_warning",
            "empty_after_trim_any",
            "first_utterance_duplicate_rate",
            "post_asr_p95_budget_ms",
            "context_reset_rate_warning",
            "validator_fail_rate",
            "pii_error_rate"
        ]

        for threshold in required_thresholds:
            assert threshold in ALERT_THRESHOLDS
            assert isinstance(ALERT_THRESHOLDS[threshold], (int, float))

    def test_session_cleanup(self):
        """Test session metrics cleanup prevents memory leaks."""
        # Add a test session
        session_id = "cleanup_test_session"
        collector = metrics_collector

        # Manually add old session
        old_session = collector.SessionMetrics(session_id)
        old_session.creation_time = time.time() - 7200  # 2 hours ago
        collector._session_metrics[session_id] = old_session

        # Force cleanup
        collector._cleanup_old_sessions()

        # Session should be cleaned up
        assert session_id not in collector._session_metrics


if __name__ == "__main__":
    print("Running hallucination metrics tests...")
    pytest.main([__file__, "-v", "--tb=short"])