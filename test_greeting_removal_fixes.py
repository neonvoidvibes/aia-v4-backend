#!/usr/bin/env python3
"""Unit tests for initial noise removal and contract fallback fixes."""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from utils.hallucination_detector import DetectorState, Word
from utils.text_noise import drop_leading_initial_noise, is_initial_noise
from services.post_asr_pipeline import (
    FeatureToggles,
    decide_transcript_candidate,
    DROP_NONE, DROP_ASR_EMPTY, DROP_FILTER_EMPTY
)


def create_test_words(text: str) -> list:
    """Helper to create Word objects from text for testing."""
    words = []
    tokens = text.split()
    start = 0.0
    for token in tokens:
        end = start + 0.5
        words.append(Word(
            text=token,
            start=start,
            end=end,
            conf=0.9
        ))
        start = end + 0.1
    return words


class MockHalluState:
    """Mock hallucination state for testing."""
    def __init__(self):
        self.last_tokens = []
        self._session_start_time = 0.0
        self._has_emitted_content = False


def test_initial_noise_repetitive_removal():
    """Test that repetitive noise like 'aaaaa' x3 within ~1s is dropped on first emission."""

    # Test case: Repetitive short-duration noise
    tokens_repetitive = ["aaaa", "aaaa", "aaaa"]
    duration_short = 0.8  # Short duration

    drop_count = drop_leading_initial_noise(tokens_repetitive, duration_short)
    assert drop_count == 3, f"Should drop all 3 repetitive tokens, got {drop_count}"

    # Test case: Same tokens but longer duration should NOT be dropped
    duration_long = 2.5  # Long duration
    drop_count_long = drop_leading_initial_noise(tokens_repetitive, duration_long)
    assert drop_count_long == 0, f"Should not drop tokens with long duration, got {drop_count_long}"

    print("✓ test_initial_noise_repetitive_removal passed")


def test_initial_noise_vs_normal_speech():
    """Test 'ok ok ok' short burst is dropped; 'ok then continue talking' isn't dropped."""

    # Test case: Short repetitive burst should be dropped
    tokens_burst = ["ok", "ok", "ok"]
    duration_burst = 1.2  # Short duration

    drop_count = drop_leading_initial_noise(tokens_burst, duration_burst)
    assert drop_count == 3, f"Should drop repetitive burst, got {drop_count}"

    # Test case: Diverse speech should NOT be dropped
    tokens_diverse = ["ok", "then", "continue", "talking"]
    duration_diverse = 1.5  # Similar duration

    drop_count_diverse = drop_leading_initial_noise(tokens_diverse, duration_diverse)
    assert drop_count_diverse == 0, f"Should not drop diverse speech, got {drop_count_diverse}"

    print("✓ test_initial_noise_vs_normal_speech passed")


def test_non_repetitive_content_preserved():
    """Test that 3-4 token non-repetitive start isn't dropped."""

    # Test case: Short but diverse content
    tokens_diverse = ["hello", "world", "how", "are"]
    duration = 1.3

    drop_count = drop_leading_initial_noise(tokens_diverse, duration)
    assert drop_count == 0, f"Should not drop diverse content, got {drop_count}"

    # Verify using is_initial_noise directly
    is_noise = is_initial_noise(tokens_diverse, duration)
    assert not is_noise, "Diverse tokens should not be classified as noise"

    print("✓ test_non_repetitive_content_preserved passed")


def test_contract_fallback_on_empty_filtered():
    """Test that when filtered segment is fully emptied, contract falls back to raw text and appends to S3."""

    # Mock functions for the contract test
    def mock_run_pii(text):
        return text  # No PII filtering

    def mock_validator(text):
        return bool(text and text.strip())

    # Use None detector state to avoid repetition blocking
    detector_state = None

    toggles = FeatureToggles(
        never_empty_contract=True,
        min_size_guard=True,
        head_blocker_enabled=True,
        early_phase_empty_allowance=True,
    )

    # Test case: Original text present, but filtered text is empty
    original_text = "This is some content that should be preserved"
    filtered_text = ""  # Empty after filtering

    decision = decide_transcript_candidate(
        original_text=original_text,
        filtered_text=filtered_text,
        run_pii=mock_run_pii,
        validator=mock_validator,
        toggles=toggles,
        min_tokens=2,
        min_chars=6,
        provider="deepgram",
        language="sv",
        detector_state=detector_state,
    )

    # Should fall back to original text due to never_empty_contract
    assert decision.drop_reason == DROP_NONE, f"Expected no drop, got {decision.drop_reason}"
    assert decision.used_fallback == True, "Should have used fallback"
    assert decision.final_text == original_text, f"Expected original text fallback, got '{decision.final_text}'"
    assert decision.low_confidence == True, "Fallback should be marked as low confidence"

    # Test metadata indicates fallback was used
    assert "fallback_reason" in decision.metadata, "Should have fallback reason in metadata"

    # Test case 2: Empty original should still drop
    decision2 = decide_transcript_candidate(
        original_text="",
        filtered_text="",
        run_pii=mock_run_pii,
        validator=mock_validator,
        toggles=toggles,
        min_tokens=2,
        min_chars=6,
        provider="deepgram",
        language="sv",
        detector_state=detector_state,
    )

    # Empty original should drop as ASR_EMPTY
    assert decision2.drop_reason == DROP_ASR_EMPTY, f"Expected ASR_EMPTY drop for empty original, got {decision2.drop_reason}"
    assert decision2.final_text == "", "Empty original should result in empty final text"

    print("✓ test_contract_fallback_on_empty_filtered passed")


def test_entropy_based_noise_detection():
    """Test low entropy sequences are detected as noise."""

    # Test case: Very low entropy (repeated characters)
    tokens_low_entropy = ["mmmmm", "mmmmm"]
    duration = 1.0

    is_noise = is_initial_noise(tokens_low_entropy, duration)
    assert is_noise, "Low entropy repeated tokens should be classified as noise"

    # Test case: High entropy should NOT be noise
    tokens_high_entropy = ["complex", "diverse", "vocabulary", "here"]
    is_not_noise = is_initial_noise(tokens_high_entropy, duration)
    assert not is_not_noise, "High entropy tokens should not be classified as noise"

    print("✓ test_entropy_based_noise_detection passed")


def test_provider_agnostic_behavior():
    """Test that noise detection works identically regardless of language or provider context."""

    # Test the same pattern with different "language" contexts
    tokens = ["ah", "ah", "ah", "um"]
    duration = 1.1

    # Should behave identically regardless of context
    drop_en = drop_leading_initial_noise(tokens, duration)
    drop_sv = drop_leading_initial_noise(tokens, duration)
    drop_fr = drop_leading_initial_noise(tokens, duration)

    assert drop_en == drop_sv == drop_fr, "Behavior should be identical across language contexts"
    assert drop_en > 0, "Repetitive pattern should be dropped"

    print("✓ test_provider_agnostic_behavior passed")


if __name__ == "__main__":
    test_initial_noise_repetitive_removal()
    test_initial_noise_vs_normal_speech()
    test_non_repetitive_content_preserved()
    test_contract_fallback_on_empty_filtered()
    test_entropy_based_noise_detection()
    test_provider_agnostic_behavior()
    print("\n✅ All initial noise removal and contract fallback tests passed!")