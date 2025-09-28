import pytest

pytest.skip("Legacy compression filter removed in append-only pipeline", allow_module_level=True)

#!/usr/bin/env python3
"""
Test the specific Swedish pattern fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.hallucination_detector import DetectorState, Word
from utils.compression_hallucination_filter import compress_filter_segment


def create_test_words(text: str, start_time: float = 0.0) -> list:
    """Helper to create Word objects from text for testing."""
    words = []
    tokens = text.split()
    current_time = start_time
    for i, token in enumerate(tokens):
        words.append(Word(
            text=token,
            start=current_time,
            end=current_time + 0.5,
            conf=0.9
        ))
        current_time += 0.5
    return words


def test_language_agnostic_fixes():
    """Test the language-agnostic compression-based fixes."""
    state = DetectorState()

    print("=== Testing Language-Agnostic Compression-Based Pattern Detection ===\n")

    # Test cases showing how compression detection works across languages
    test_cases = [
        {
            "name": "First occurrence (establishes baseline)",
            "text": "hello world this is a test of the system today"
        },
        {
            "name": "Cross-segment repetition (should be adaptively trimmed)",
            "text": "hello world and now we have different content after the repeated start"
        },
        {
            "name": "Another repetition (should be trimmed)",
            "text": "hello world but this time with completely new ending content here"
        },
        {
            "name": "High compression ratio (should be caught)",
            "text": "test test test test this pattern should be detected automatically"
        },
        {
            "name": "Normal content (should pass)",
            "text": "completely different content with no repetitive patterns at all"
        },
        {
            "name": "Swedish example (language-agnostic approach)",
            "text": "åh ni har gjort något nytt som är helt annorlunda från tidigare"
        },
        {
            "name": "Swedish repetition (should be detected by compression)",
            "text": "åh ni och denna ballong situation kommer att identifieras automatiskt"
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"Test {i+1}: {case['name']}")
        print(f"Input:  {case['text']}")

        # Convert to Word objects
        words = create_test_words(case['text'])

        # Apply compression filtering (language-agnostic)
        filtered_words, reason, cut_count = compress_filter_segment(
            words, state, provider="deepgram", language="unknown"
        )

        # Results
        filtered_text = " ".join([w.text for w in filtered_words])
        print(f"Output: {filtered_text}" if filtered_text else "  [DROPPED]")
        print(f"Reason: {reason}")
        print(f"Cut:    {cut_count} words")

        # Check compression stats
        if hasattr(state, 'compression_detector'):
            stats = state.compression_detector.get_recent_compression_stats()
            print(f"Stats:  avg_ratio={stats.get('avg_compression_ratio', 0):.2f}")

        print()

    print("=== Test Complete ===")


if __name__ == "__main__":
    test_language_agnostic_fixes()