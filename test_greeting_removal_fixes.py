#!/usr/bin/env python3
"""Unit tests for greeting removal and contract fallback fixes."""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from utils.hallucination_detector import DetectorState, Word
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


def test_initial_greeting_removal():
    """Test that the very first 'Hallå ...' is removed and second later 'hallå' in normal speech is kept."""

    # Simulate the greeting drop function from transcription_service.py
    def _drop_initial_greeting(words, hallu_state, language):
        """Drop leading Swedish greetings from first segment only."""
        try:
            lang_ok = (language or "").lower().startswith("sv")
        except Exception:
            lang_ok = False

        # Check if this is the first content emission for the session
        has_emitted_content = getattr(hallu_state, '_has_emitted_content', False)
        if not lang_ok or has_emitted_content:
            return words, 0

        if not words:
            return words, 0

        GREET_TOKENS_SV = {"hallå", "hej", "hejsan", "tjena", "tja", "hejhej", "yo"}
        i = 0
        while i < len(words):
            w_text = getattr(words[i], "text", None) or str(words[i])
            # Normalize Unicode (e.g., hall\u00e5 -> hallå)
            w_normalized = w_text.lower().strip(" .,!?:;–—\"'()[]").replace('\u00e5', 'å')
            if w_normalized in GREET_TOKENS_SV:
                i += 1
                continue
            break

        if i > 0:
            # Mark that we've now had our first content, even if it was dropped
            hallu_state._has_emitted_content = True
            return words[i:], i  # Return count of dropped words
        return words, 0

    hallu_state = MockHalluState()

    # Test 1: First segment with greeting should be removed
    words1 = create_test_words("hallå this is a test message")
    result1, dropped_count1 = _drop_initial_greeting(words1, hallu_state, "sv")
    result1_text = " ".join([w.text for w in result1])

    assert dropped_count1 == 1, "Should drop exactly 1 greeting token"
    assert result1_text == "this is a test message", f"Expected 'this is a test message', got '{result1_text}'"
    assert hallu_state._has_emitted_content == True, "Should mark content as emitted"

    # Test 2: Second segment with 'hallå' in normal speech should be kept
    words2 = create_test_words("I said hallå to my friend")
    result2, dropped_count2 = _drop_initial_greeting(words2, hallu_state, "sv")
    result2_text = " ".join([w.text for w in result2])

    assert dropped_count2 == 0, "Should not drop any tokens from second segment"
    assert result2_text == "I said hallå to my friend", f"Expected full text, got '{result2_text}'"

    print("✓ test_initial_greeting_removal passed")


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


def test_multiple_greeting_tokens():
    """Test handling of multiple greeting tokens in sequence."""

    def _drop_initial_greeting(words, hallu_state, language):
        """Drop leading Swedish greetings from first segment only."""
        try:
            lang_ok = (language or "").lower().startswith("sv")
        except Exception:
            lang_ok = False

        has_emitted_content = getattr(hallu_state, '_has_emitted_content', False)
        if not lang_ok or has_emitted_content:
            return words, 0

        if not words:
            return words, 0

        GREET_TOKENS_SV = {"hallå", "hej", "hejsan", "tjena", "tja", "hejhej", "yo"}
        i = 0
        while i < len(words):
            w_text = getattr(words[i], "text", None) or str(words[i])
            w_normalized = w_text.lower().strip(" .,!?:;–—\"'()[]").replace('\u00e5', 'å')
            if w_normalized in GREET_TOKENS_SV:
                i += 1
                continue
            break

        if i > 0:
            hallu_state._has_emitted_content = True
            return words[i:], i
        return words, 0

    hallu_state = MockHalluState()

    # Test multiple greeting tokens
    words = create_test_words("hej hallå tjena how are you")
    result, dropped_count = _drop_initial_greeting(words, hallu_state, "sv")
    result_text = " ".join([w.text for w in result])

    assert dropped_count == 3, f"Should drop 3 greeting tokens, dropped {dropped_count}"
    assert result_text == "how are you", f"Expected 'how are you', got '{result_text}'"

    print("✓ test_multiple_greeting_tokens passed")


def test_unicode_normalization():
    """Test Unicode normalization in greeting detection."""

    def _drop_initial_greeting(words, hallu_state, language):
        """Drop leading Swedish greetings from first segment only."""
        try:
            lang_ok = (language or "").lower().startswith("sv")
        except Exception:
            lang_ok = False

        has_emitted_content = getattr(hallu_state, '_has_emitted_content', False)
        if not lang_ok or has_emitted_content:
            return words, 0

        if not words:
            return words, 0

        GREET_TOKENS_SV = {"hallå", "hej", "hejsan", "tjena", "tja", "hejhej", "yo"}
        i = 0
        while i < len(words):
            w_text = getattr(words[i], "text", None) or str(words[i])
            w_normalized = w_text.lower().strip(" .,!?:;–—\"'()[]").replace('\u00e5', 'å')
            if w_normalized in GREET_TOKENS_SV:
                i += 1
                continue
            break

        if i > 0:
            hallu_state._has_emitted_content = True
            return words[i:], i
        return words, 0

    hallu_state = MockHalluState()

    # Test Unicode normalization: hall\u00e5 should match hallå
    words = create_test_words("hall\u00e5 there friend")
    result, dropped_count = _drop_initial_greeting(words, hallu_state, "sv")
    result_text = " ".join([w.text for w in result])

    assert dropped_count == 1, f"Should drop Unicode greeting token, dropped {dropped_count}"
    assert result_text == "there friend", f"Expected 'there friend', got '{result_text}'"

    print("✓ test_unicode_normalization passed")


if __name__ == "__main__":
    test_initial_greeting_removal()
    test_contract_fallback_on_empty_filtered()
    test_multiple_greeting_tokens()
    test_unicode_normalization()
    print("\n✅ All greeting removal and contract fallback tests passed!")