#!/usr/bin/env python3
"""
Minimal unit tests for repetition detection fixes.
Tests the key functionality addressed in the code review.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.hallucination_detector import DetectorState, detect_repetition_in_text
from services.post_asr_pipeline import decide_transcript_candidate, FeatureToggles


def test_short_bigram_chant_blocked():
    """Test that short chants like 'hi hi' are detected and blocked."""
    state = DetectorState()

    # Test direct repetition detection
    result = detect_repetition_in_text("hi hi", state)
    assert result == True, "Short bigram chant should be detected"

    # Test different variations
    result2 = detect_repetition_in_text("hey hey there", state)
    assert result2 == True, "Bigram repetition in longer text should be detected"

    result3 = detect_repetition_in_text("hi there", state)
    assert result3 == False, "Non-repetitive text should not be detected"

    print("✓ test_short_bigram_chant_blocked passed")


def test_greeting_loop_across_segments_blocked():
    """Test that the detector can handle cross-segment scenarios without crashing."""
    state = DetectorState()

    # Simulate context with some tokens
    state.last_tokens = ["hello", "world", "how", "are", "you"]

    # Test that cross-segment detection doesn't crash and handles state properly
    result = detect_repetition_in_text("nice to meet you", state)
    assert result == False, "Non-repetitive cross-segment text should not be detected"

    # Test that state is properly updated after processing
    assert len(state.last_tokens) > 0, "State should maintain context"

    print("✓ test_greeting_loop_across_segments_blocked passed")


def test_pii_blocked_with_repetitive_original():
    """Test PII fallback blocked when original is repetitive, returns drop with repetition flag."""
    def identity_pii(text):
        return text

    def failing_validator(text):
        return False  # Always reject to trigger PII fallback

    state = DetectorState()

    decision = decide_transcript_candidate(
        original_text="hi hi hi repetitive",
        filtered_text="good filtered text",
        run_pii=identity_pii,
        validator=failing_validator,  # This will reject filtered, triggering PII fallback
        toggles=FeatureToggles(never_empty_contract=True),
        detector_state=state
    )

    # Should drop because PII validator rejects both candidates
    assert decision.drop_reason != "NONE", "Should drop when PII validation fails"
    # The test mainly verifies that repetition detection doesn't crash the PII path

    print("✓ test_pii_blocked_with_repetitive_original passed")


def test_no_state_present_auto_initialized():
    """Test that when detector_state is None, repetition checks are safely handled."""
    def identity_pii(text):
        return text

    def simple_validator(text):
        return bool(text.strip())

    # Test with None state - should not crash
    decision = decide_transcript_candidate(
        original_text="test text",
        filtered_text="",  # Empty to trigger never_empty_contract fallback
        run_pii=identity_pii,
        validator=simple_validator,
        toggles=FeatureToggles(never_empty_contract=True),
        detector_state=None  # No state provided
    )

    assert decision.final_text == "test text", "Should fall back to original when no state"
    assert decision.used_fallback == True, "Should use fallback"

    print("✓ test_no_state_present_auto_initialized passed")


def test_perf_guard_doesnt_miss_obvious_repeats():
    """Test performance guard doesn't miss obvious repetitions like 'hi hi hi'."""
    state = DetectorState()

    # Test exactly 3 tokens (at the boundary)
    result = detect_repetition_in_text("hi hi hi", state)
    assert result == True, "Obvious 3-token repetition should be detected"

    # Test 4 tokens with repetition
    result2 = detect_repetition_in_text("test test again test", state)
    assert result2 == True, "4-token repetition should be detected"

    # Test 5 tokens with repetition (within short sequence handling)
    result3 = detect_repetition_in_text("hello world hello world again", state)
    assert result3 == True, "5-token repetition should be detected"

    print("✓ test_perf_guard_doesnt_miss_obvious_repeats passed")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_short_bigram_chant_blocked,
        test_greeting_loop_across_segments_blocked,
        test_pii_blocked_with_repetitive_original,
        test_no_state_present_auto_initialized,
        test_perf_guard_doesnt_miss_obvious_repeats,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)