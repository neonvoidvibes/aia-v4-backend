"""
Compression ratio-based repetition detection for ASR transcripts.
Based on OpenAI Whisper's proven approach for hallucination detection.
"""

import zlib
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import threading


@dataclass
class CompressionDetectorConfig:
    """Configuration for compression-based repetition detection."""
    # Primary threshold - segments with higher ratio are considered repetitive
    # Research shows 1.35 is effective for catching subtle repetitions
    compression_ratio_threshold: float = 1.35

    # Hard threshold - segments above this are always dropped
    compression_ratio_hard_limit: float = 2.4

    # Minimum segment length to check (avoid false positives on short text)
    min_segment_length: int = 10

    # Rolling window for context-aware detection
    context_window_size: int = 5

    # Threshold for detecting repetitive patterns across segments
    # Tuned to catch "hejsan hallå" patterns without false positives
    cross_segment_threshold: float = 1.15


@dataclass
class SegmentStats:
    """Statistics for a processed segment."""
    text: str
    compression_ratio: float
    timestamp: float
    is_repetitive: bool
    retry_count: int = 0


class CompressionRepetitionDetector:
    """
    Detects repetitive segments using zlib compression ratio.

    This approach is proven effective in Whisper ASR systems:
    - High compression ratio indicates repetitive text
    - Simple, fast, and language-agnostic
    - Used by OpenAI, Faster-Whisper, and other production systems
    """

    def __init__(self, config: Optional[CompressionDetectorConfig] = None):
        self.config = config or CompressionDetectorConfig()
        self.recent_segments: deque = deque(maxlen=self.config.context_window_size)
        self._lock = threading.RLock()

    def calculate_compression_ratio(self, text: str) -> float:
        """
        Calculate zlib compression ratio for text.
        Higher ratio = more repetitive content.
        """
        if not text or len(text) < 2:
            return 0.0

        try:
            # Encode to bytes for compression
            text_bytes = text.encode('utf-8')
            compressed = zlib.compress(text_bytes)

            # Ratio = original_size / compressed_size
            # Higher ratio means better compression (more repetitive)
            ratio = len(text_bytes) / len(compressed)
            return ratio
        except Exception:
            # If compression fails, assume no repetition
            return 1.0

    def is_segment_repetitive(self, text: str) -> Tuple[bool, float, str]:
        """
        Determine if a segment is repetitive based on compression ratio.
        Returns (is_repetitive, compression_ratio, reason).
        """
        if not text or len(text) < self.config.min_segment_length:
            return False, 1.0, "too_short"

        compression_ratio = self.calculate_compression_ratio(text)

        # Hard limit check - always drop
        if compression_ratio > self.config.compression_ratio_hard_limit:
            return True, compression_ratio, "hard_limit_exceeded"

        # Primary threshold check
        if compression_ratio > self.config.compression_ratio_threshold:
            return True, compression_ratio, "threshold_exceeded"

        return False, compression_ratio, "normal"

    def detect_cross_segment_repetition(self, text: str) -> Tuple[bool, str]:
        """
        Check if current segment repeats content from recent segments.
        This catches cases where individual segments aren't repetitive,
        but the pattern repeats across segments (like "hejsan hallå").
        """
        with self._lock:
            if not self.recent_segments:
                return False, "no_history"

            # Combine recent segments for context
            recent_text = " ".join([seg.text for seg in self.recent_segments])
            combined_text = recent_text + " " + text

            # Check if combined text is more compressible than expected
            combined_ratio = self.calculate_compression_ratio(combined_text)
            individual_avg_ratio = sum([seg.compression_ratio for seg in self.recent_segments]) / len(self.recent_segments)

            # If combined text compresses much better than individual segments,
            # it indicates repetition across segments
            if combined_ratio > individual_avg_ratio * self.config.cross_segment_threshold:
                return True, "cross_segment_pattern"

            return False, "normal"

    def process_segment(self, text: str, provider: str = "unknown") -> Dict:
        """
        Process a segment and return detection results.
        This is the main entry point for the detector.
        """
        timestamp = time.monotonic()

        # Check individual segment repetition
        is_repetitive, compression_ratio, reason = self.is_segment_repetitive(text)

        # Check cross-segment repetition if individual check passes
        cross_segment_repetitive = False
        cross_segment_reason = "not_checked"

        if not is_repetitive:
            cross_segment_repetitive, cross_segment_reason = self.detect_cross_segment_repetition(text)
            if cross_segment_repetitive:
                is_repetitive = True
                reason = cross_segment_reason

        # Create segment stats
        segment_stats = SegmentStats(
            text=text,
            compression_ratio=compression_ratio,
            timestamp=timestamp,
            is_repetitive=is_repetitive
        )

        # Update rolling window
        with self._lock:
            self.recent_segments.append(segment_stats)

        return {
            "is_repetitive": is_repetitive,
            "compression_ratio": compression_ratio,
            "reason": reason,
            "cross_segment_repetitive": cross_segment_repetitive,
            "cross_segment_reason": cross_segment_reason,
            "timestamp": timestamp,
            "provider": provider,
            "should_retry": compression_ratio > self.config.compression_ratio_threshold and compression_ratio < self.config.compression_ratio_hard_limit,
            "should_drop": compression_ratio > self.config.compression_ratio_hard_limit
        }

    def get_recent_compression_stats(self) -> Dict:
        """Get statistics about recent segments for monitoring."""
        with self._lock:
            if not self.recent_segments:
                return {"count": 0}

            ratios = [seg.compression_ratio for seg in self.recent_segments]
            repetitive_count = sum(1 for seg in self.recent_segments if seg.is_repetitive)

            return {
                "count": len(self.recent_segments),
                "avg_compression_ratio": sum(ratios) / len(ratios),
                "max_compression_ratio": max(ratios),
                "repetitive_segments": repetitive_count,
                "repetitive_percentage": repetitive_count / len(self.recent_segments) * 100
            }

    def reset_context(self):
        """Reset the detector context (useful for new sessions)."""
        with self._lock:
            self.recent_segments.clear()


def test_compression_detector():
    """Test the compression detector with example patterns."""
    detector = CompressionRepetitionDetector()

    # Test with actual patterns from the Swedish transcript
    test_cases = [
        # Normal speech from transcript
        "du provade dig för en chans sen behöver vi fortsätta läsa transkriptet och",

        # The problematic "hejsan hallå" pattern
        "hejsan hallå nu provar vi inspelning igen hallucinering det har gått 10 sekunder vi behöver 15",

        # Another "hejsan hallå" occurrence
        "hejsan hallå du provar jag blir ganska irriterad om det inte funkar ja titta det funkar det",

        # Third "hejsan hallå" pattern
        "hejsan hallå det prov verkar som att den kuttade mer än den borde det var inget bra nej det fortsätter",

        # "nu provar" pattern repetition
        "nu provar så det borde inte vara något problem egentligen verkar som att det kanske är bättre nej det är inte bättre",

        # More repetitive pattern
        "nu provar ni börjar bli presterande tack och hej",

        # Highly repetitive for comparison
        "test test test test test test test test test test test test",

        # Normal Swedish speech
        "Men det var en bra dag idag och jag hoppas vi kan träffas snart igen"
    ]

    print("=== Compression Ratio Repetition Detection Test ===")
    for i, text in enumerate(test_cases):
        result = detector.process_segment(text)
        print(f"\nTest {i+1}: {text[:50]}...")
        print(f"  Compression Ratio: {result['compression_ratio']:.2f}")
        print(f"  Is Repetitive: {result['is_repetitive']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Should Retry: {result['should_retry']}")
        print(f"  Should Drop: {result['should_drop']}")

    print(f"\n=== Recent Stats ===")
    stats = detector.get_recent_compression_stats()
    print(f"Segments processed: {stats['count']}")
    print(f"Average compression ratio: {stats.get('avg_compression_ratio', 0):.2f}")
    print(f"Repetitive percentage: {stats.get('repetitive_percentage', 0):.1f}%")


if __name__ == "__main__":
    test_compression_detector()