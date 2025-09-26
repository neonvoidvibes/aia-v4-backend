#!/usr/bin/env python3
"""
Targeted tests for the refactored hallucination filtering system.
Tests the key contract guarantees and overlap detection logic.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from utils.hallucination_detector import DetectorState, Word, maybe_trim_repetition
from services.post_asr_pipeline import (
    FeatureToggles,
    decide_transcript_candidate,
    DROP_NONE, DROP_ASR_EMPTY, DROP_FILTER_EMPTY, DROP_PII_ERROR
)


def simple_validator(text: str) -> bool:
    """Simple validator that rejects empty/short text."""
    return bool(text and text.strip() and len(text.strip()) >= 2)


def identity_pii(text: str) -> str:
    """Identity PII function for testing."""
    return text


def failing_pii(text: str) -> str:
    """PII function that raises an exception."""
    raise RuntimeError("PII service unavailable")


class TestOverlapTrimming:
    """Test overlap detection and trimming thresholds."""

    def test_no_overlap_with_empty_context(self):
        """No trimming should occur when context is empty."""
        state = DetectorState()
        words = [
            Word("Hello", 0.0, 1.0, 0.9),
            Word("world", 1.0, 2.0, 0.9),
        ]

        result_words, reason, cut_count = maybe_trim_repetition(words, state)

        assert len(result_words) == 2
        assert reason is None
        assert cut_count == 0
        # Context should be updated
        assert len(state.last_tokens) == 2

    def test_overlap_detection_with_context(self):
        """Test overlap detection when context contains overlapping tokens."""
        state = DetectorState()
        # Pre-populate context
        state.last_tokens = ["hello", "world", "this", "is"]

        words = [
            Word("This", 0.0, 1.0, 0.9),
            Word("is", 1.0, 2.0, 0.9),
            Word("new", 2.0, 3.0, 0.9),
            Word("content", 3.0, 4.0, 0.9),
        ]

        # Use low threshold to trigger overlap detection
        result_words, reason, cut_count = maybe_trim_repetition(
            words, state, min_overlap_tokens=1, min_segment_tokens=1
        )

        # Should trim overlapping "This is" (2 words)
        assert len(result_words) == 2  # "new", "content"
        assert result_words[0].text == "new"
        assert result_words[1].text == "content"
        assert reason == "trimmed_overlap:2"
        assert cut_count == 2

    def test_insufficient_overlap_threshold(self):
        """Test that small overlaps below threshold are not trimmed."""
        state = DetectorState()
        state.last_tokens = ["previous", "context"]

        words = [
            Word("context", 0.0, 1.0, 0.9),  # Only 1 word overlap
            Word("new", 1.0, 2.0, 0.9),
            Word("words", 2.0, 3.0, 0.9),
        ]

        # With default min_overlap_tokens=12, this won't trigger
        result_words, reason, cut_count = maybe_trim_repetition(words, state)

        assert len(result_words) == 3  # No trimming
        assert reason is None
        assert cut_count == 0

    def test_context_updates_on_trim(self):
        """Test that context is properly updated with kept tokens."""
        state = DetectorState()
        state.last_tokens = ["hello"] * 15  # Create sufficient context

        words = [Word("hello", 0.0, 1.0, 0.9)] * 15 + [Word("new", 15.0, 16.0, 0.9)]

        result_words, reason, cut_count = maybe_trim_repetition(words, state)

        # Should keep only "new"
        assert len(result_words) == 1
        assert result_words[0].text == "new"
        assert reason.startswith("trimmed_overlap:")
        # Context should contain "new"
        assert "new" in state.last_tokens

    def test_context_updates_on_no_trim(self):
        """Test that context is updated even when no trimming occurs."""
        state = DetectorState()
        original_context_len = len(state.last_tokens)

        words = [
            Word("completely", 0.0, 1.0, 0.9),
            Word("new", 1.0, 2.0, 0.9),
            Word("content", 2.0, 3.0, 0.9),
        ]

        result_words, reason, cut_count = maybe_trim_repetition(words, state)

        assert len(result_words) == 3
        assert reason is None
        # Context should grow
        assert len(state.last_tokens) > original_context_len
        assert "completely" in state.last_tokens
        assert "new" in state.last_tokens
        assert "content" in state.last_tokens


class TestPostASRContract:
    """Test post-ASR decision contract enforcement."""

    def test_asr_empty_results_in_drop(self):
        """Empty ASR input should result in ASR_EMPTY drop."""
        decision = decide_transcript_candidate(
            original_text="",
            filtered_text="anything",  # Should be ignored
            run_pii=identity_pii,
            validator=simple_validator,
            toggles=FeatureToggles(never_empty_contract=True, min_size_guard=True)
        )

        assert decision.final_text == ""
        assert decision.drop_reason == DROP_ASR_EMPTY
        assert not decision.used_fallback
        assert not decision.pii_pass

    def test_never_empty_contract_fallback(self):
        """When filtered text is empty, should fallback to original."""
        decision = decide_transcript_candidate(
            original_text="Original text here",
            filtered_text="",
            run_pii=identity_pii,
            validator=simple_validator,
            toggles=FeatureToggles(never_empty_contract=True, min_size_guard=True)
        )

        assert decision.final_text == "Original text here"
        assert decision.drop_reason == DROP_NONE
        assert decision.used_fallback
        assert decision.low_confidence
        assert decision.pii_pass

    def test_min_size_guard_fallback(self):
        """Small filtered text should fallback to original."""
        decision = decide_transcript_candidate(
            original_text="This is the original longer text",
            filtered_text="x",  # Too small
            run_pii=identity_pii,
            validator=simple_validator,
            toggles=FeatureToggles(never_empty_contract=True, min_size_guard=True)
        )

        assert decision.final_text == "This is the original longer text"
        assert decision.used_fallback
        assert decision.low_confidence

    def test_disabled_contracts_allow_drops(self):
        """With contracts disabled, empty filtered text should drop."""
        decision = decide_transcript_candidate(
            original_text="Original text",
            filtered_text="",
            run_pii=identity_pii,
            validator=simple_validator,
            toggles=FeatureToggles(never_empty_contract=False, min_size_guard=False)
        )

        assert decision.final_text == ""
        assert decision.drop_reason == DROP_FILTER_EMPTY
        assert not decision.used_fallback

    def test_pii_error_results_in_drop(self):
        """PII processing errors should result in PII_ERROR drop."""
        decision = decide_transcript_candidate(
            original_text="Some text",
            filtered_text="Some text",
            run_pii=failing_pii,
            validator=simple_validator,
            toggles=FeatureToggles(never_empty_contract=True, min_size_guard=True)
        )

        assert decision.final_text == ""
        assert decision.drop_reason == DROP_PII_ERROR
        assert not decision.pii_pass

    def test_successful_path_returns_text(self):
        """Successful processing should return processed text."""
        decision = decide_transcript_candidate(
            original_text="Original text here",
            filtered_text="Filtered text here",
            run_pii=identity_pii,
            validator=simple_validator,
            toggles=FeatureToggles(never_empty_contract=True, min_size_guard=True)
        )

        assert decision.final_text == "Filtered text here"
        assert decision.drop_reason == DROP_NONE
        assert not decision.used_fallback
        assert not decision.low_confidence
        assert decision.pii_pass


class TestParameterizedTunables:
    """Test that overlap detection can be tuned."""

    def test_custom_overlap_threshold(self):
        """Test override of overlap threshold."""
        state = DetectorState()
        state.last_tokens = ["hello", "world"]

        words = [
            Word("hello", 0.0, 1.0, 0.9),
            Word("world", 1.0, 2.0, 0.9),
            Word("new", 2.0, 3.0, 0.9),
        ]

        # With low threshold, should trigger trimming
        result_words, reason, cut_count = maybe_trim_repetition(
            words, state, min_overlap_tokens=1, min_segment_tokens=1
        )

        assert len(result_words) == 1  # Should trim 2 words
        assert result_words[0].text == "new"
        assert reason == "trimmed_overlap:2"

    def test_custom_context_limit(self):
        """Test override of context limit."""
        state = DetectorState()
        state.last_tokens = ["word"] * 100  # Very long context

        words = [Word("word", 0.0, 1.0, 0.9)] * 5 + [Word("new", 5.0, 6.0, 0.9)]

        # With small context limit, might not find overlap
        result_words, reason, cut_count = maybe_trim_repetition(
            words, state, max_context_tokens=2, min_overlap_tokens=1
        )

        # Should still work with limited context
        assert "new" in [w.text for w in result_words]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])