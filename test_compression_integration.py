#!/usr/bin/env python3
"""
Test compression-based hallucination filtering with Swedish transcript patterns.
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


def test_compression_filtering():
    """Test compression-based filtering with Swedish transcript patterns."""
    state = DetectorState()

    # Test cases from the actual Swedish transcript
    test_cases = [
        {
            "name": "First hejsan hallå (should pass)",
            "text": "hejsan hallå nu provar vi inspelning igen hallucinering det har gått 10 sekunder vi behöver 15"
        },
        {
            "name": "Second hejsan hallå (should be filtered)",
            "text": "hejsan hallå du provar jag blir ganska irriterad om det inte funkar ja titta det funkar det"
        },
        {
            "name": "Third hejsan hallå (should be filtered)",
            "text": "hejsan hallå det prov verkar som att den kuttade mer än den borde det var inget bra nej det fortsätter"
        },
        {
            "name": "Normal Swedish speech (should pass)",
            "text": "du provade dig för en chans sen behöver vi fortsätta läsa transkriptet och"
        },
        {
            "name": "nu provar pattern (should be filtered after repetition)",
            "text": "nu provar så det borde inte vara något problem egentligen verkar som att det kanske är bättre nej det är inte bättre"
        },
        {
            "name": "Another nu provar (should be filtered)",
            "text": "nu provar ni börjar bli presterande tack och hej"
        },
        {
            "name": "High repetition (should be dropped)",
            "text": "test test test test test test test test test test test test"
        }
    ]

    print("=== Compression-Based Hallucination Filter Test ===")
    print("Testing with Swedish transcript patterns...\n")

    for i, case in enumerate(test_cases):
        print(f"Test {i+1}: {case['name']}")
        print(f"Input: {case['text'][:60]}...")

        # Convert to Word objects
        words = create_test_words(case['text'])

        # Apply compression filtering
        filtered_words, reason, cut_count = compress_filter_segment(
            words, state, provider="deepgram", language="swedish"
        )

        # Results
        filtered_text = " ".join([w.text for w in filtered_words])
        print(f"Output: {filtered_text[:60]}..." if filtered_text else "  [DROPPED]")
        print(f"Reason: {reason}")
        print(f"Words cut: {cut_count}")

        # Compression stats
        stats = state.compression_detector.get_recent_compression_stats()
        print(f"Compression stats: avg_ratio={stats.get('avg_compression_ratio', 0):.2f}, repetitive={stats.get('repetitive_percentage', 0):.1f}%")
        print()

    print("=== Final Statistics ===")
    final_stats = state.compression_detector.get_recent_compression_stats()
    print(f"Total segments processed: {final_stats.get('count', 0)}")
    print(f"Average compression ratio: {final_stats.get('avg_compression_ratio', 0):.2f}")
    print(f"Repetitive segments: {final_stats.get('repetitive_percentage', 0):.1f}%")


if __name__ == "__main__":
    test_compression_filtering()