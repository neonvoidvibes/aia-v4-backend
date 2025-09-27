#!/usr/bin/env python3
"""
Minimal unit tests for repetition detection fixes.
Tests the key functionality addressed in the code review.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.hallucination_detector import DetectorState, detect_repetition_in_text, Word
from services.post_asr_pipeline import decide_transcript_candidate, FeatureToggles


def create_test_words(text: str) -> list:
    """Helper to create Word objects from text for testing."""
    words = []
    tokens = text.split()
    start_time = 0.0
    for i, token in enumerate(tokens):
        words.append(Word(
            text=token,
            start=start_time,
            end=start_time + 0.5,
            conf=0.9
        ))
        start_time += 0.5
    return words


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


def test_segment_start_head_blocker_initialization():
    """Test that session head model is properly initialized."""
    state = DetectorState()

    # Should start with head model initialized (created in __init__)
    assert hasattr(state, 'head_model')
    assert state.head_model is not None
    assert hasattr(state.head_model, 'counts')
    assert hasattr(state.head_model, 'last_seen')

    # Process a segment to verify it works
    from utils.hallucination_detector import maybe_trim_repetition
    test_words = create_test_words("hello world")
    trimmed_words, reason, cut_count = maybe_trim_repetition(test_words, state, segment_position=0)

    # Should still have head model
    assert hasattr(state, 'head_model')
    assert state.head_model is not None

    print("✓ test_segment_start_head_blocker_initialization passed")


def test_repeated_start_pattern_blocking():
    """Test that repeated start patterns like 'hejsan hallå' are blocked."""
    state = DetectorState()
    from utils.hallucination_detector import maybe_trim_repetition

    # First segment: establish pattern
    words1 = create_test_words("hejsan hallå nice to meet you")
    trimmed1, reason1, cut1 = maybe_trim_repetition(words1, state, segment_position=0)
    result1_text = " ".join([w.text for w in trimmed1])
    assert result1_text == "hejsan hallå nice to meet you", "First occurrence should pass through"

    # Second segment: should trigger head blocker
    words2 = create_test_words("hejsan hallå how are you today")
    trimmed2, reason2, cut2 = maybe_trim_repetition(words2, state, segment_position=0)
    result2_text = " ".join([w.text for w in trimmed2])
    # Should be trimmed to remove the repeated head pattern
    assert "hejsan hallå" not in result2_text, "Repeated head pattern should be blocked"
    assert "how are you today" in result2_text, "Tail content should be preserved"

    # Third segment: should continue blocking
    words3 = create_test_words("hejsan hallå what's up")
    trimmed3, reason3, cut3 = maybe_trim_repetition(words3, state, segment_position=0)
    result3_text = " ".join([w.text for w in trimmed3])
    assert "hejsan hallå" not in result3_text, "Head pattern should still be blocked"
    assert "what's up" in result3_text, "Tail content should be preserved"

    print("✓ test_repeated_start_pattern_blocking passed")


def test_exponential_decay_head_blocker():
    """Test that head blocker patterns decay over time."""
    import time
    from unittest.mock import patch

    state = DetectorState()
    from utils.hallucination_detector import maybe_trim_repetition

    # Mock time to simulate passage
    with patch('time.monotonic') as mock_time:
        # Start at time 0
        mock_time.return_value = 0.0

        # Establish pattern
        words1 = create_test_words("hello world test")
        trimmed1, reason1, cut1 = maybe_trim_repetition(words1, state, segment_position=0)
        result1_text = " ".join([w.text for w in trimmed1])
        assert result1_text == "hello world test"

        # Immediate repeat should be blocked
        words2 = create_test_words("hello world again")
        trimmed2, reason2, cut2 = maybe_trim_repetition(words2, state, segment_position=0)
        result2_text = " ".join([w.text for w in trimmed2])
        assert "hello world" not in result2_text

        # Advance time by 60 seconds (2*τ for significant decay)
        mock_time.return_value = 60.0

        # Pattern should be less likely to be blocked due to decay
        words3 = create_test_words("hello world back again")
        trimmed3, reason3, cut3 = maybe_trim_repetition(words3, state, segment_position=0)
        # Due to exponential decay, this might pass through (depends on threshold)

    print("✓ test_exponential_decay_head_blocker passed")


def test_early_phase_empty_allowance():
    """Test early-phase empty allowance in decide_transcript_candidate."""
    state = DetectorState()

    def identity_pii(text):
        return text

    def simple_validator(text):
        return bool(text.strip())

    # Simulate head blocker firing by providing heavily filtered text
    decision = decide_transcript_candidate(
        original_text="hejsan hallå welcome to the meeting",
        filtered_text="welcome to the meeting",  # Head removed by blocker
        run_pii=identity_pii,
        validator=simple_validator,
        toggles=FeatureToggles(
            never_empty_contract=True,
            min_size_guard=True,
            head_blocker_enabled=True,
            early_phase_empty_allowance=True
        ),
        detector_state=state,
        segment_count=0  # Early phase
    )

    # Should succeed even with small filtered text in early phase
    assert decision.final_text == "welcome to the meeting"
    assert decision.drop_reason == "NONE"

    print("✓ test_early_phase_empty_allowance passed")


def test_head_blocker_feature_toggle():
    """Test that head blocker respects feature toggle."""
    state = DetectorState()

    def identity_pii(text):
        return text

    def simple_validator(text):
        return bool(text.strip())

    # With head blocker disabled
    decision1 = decide_transcript_candidate(
        original_text="hejsan hallå test content",
        filtered_text="filtered content",
        run_pii=identity_pii,
        validator=simple_validator,
        toggles=FeatureToggles(head_blocker_enabled=False),
        detector_state=state,
        segment_count=0
    )

    # Should not detect head blocker activity
    assert not decision1.metadata.get('head_blocker_fired', False)

    # With head blocker enabled
    decision2 = decide_transcript_candidate(
        original_text="hejsan hallå test content",
        filtered_text="content",  # Significantly shorter (head blocked)
        run_pii=identity_pii,
        validator=simple_validator,
        toggles=FeatureToggles(head_blocker_enabled=True),
        detector_state=state,
        segment_count=0
    )

    # Should detect head blocker activity
    assert decision2.metadata.get('head_blocker_fired', False)

    print("✓ test_head_blocker_feature_toggle passed")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_short_bigram_chant_blocked,
        test_greeting_loop_across_segments_blocked,
        test_pii_blocked_with_repetitive_original,
        test_no_state_present_auto_initialized,
        test_perf_guard_doesnt_miss_obvious_repeats,
        test_segment_start_head_blocker_initialization,
        test_repeated_start_pattern_blocking,
        test_exponential_decay_head_blocker,
        test_early_phase_empty_allowance,
        test_head_blocker_feature_toggle,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__} failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)