"""
Compression-based hallucination filter for ASR transcripts.
Integrates with existing hallucination detector as a pre-filter.
"""

import logging
from typing import List, Optional, Tuple, Dict

# Setup logger
logger = logging.getLogger(__name__)

try:
    from .compression_repetition_detector import CompressionRepetitionDetector, CompressionDetectorConfig
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

try:
    from .hallucination_detector import DetectorState, Word, maybe_trim_repetition
    MAIN_DETECTOR_AVAILABLE = True
except ImportError:
    MAIN_DETECTOR_AVAILABLE = False


def compress_filter_segment(
    words: List[Word],
    state: DetectorState,
    provider: str = "unknown",
    language: str = "unknown",
    ctx: 'DetectorContext' = None
) -> Tuple[List[Word], Optional[str], int]:
    """
    Apply compression-based filtering before main hallucination detection.

    This function:
    1. Converts words to text and checks compression ratio
    2. If highly repetitive, drops entire segment
    3. If moderately repetitive, applies aggressive trimming
    4. Otherwise, passes through to main detector

    Returns (filtered_words, reason, cut_word_count)
    """
    if not words:
        return words, "empty_input", 0

    # Convert words to text for compression analysis
    segment_text = " ".join([w.text for w in words])

    # Use compression detector if available
    if hasattr(state, 'compression_detector') and state.compression_detector:
        try:
            result = state.compression_detector.process_segment(segment_text, provider=provider)

            # Hard drop: very high compression ratio
            if result['should_drop']:
                logger.info(f"Compression filter: dropping segment (ratio={result['compression_ratio']:.2f})")
                return [], "compression_hard_drop", len(words)

            # Aggressive trim: moderate compression ratio
            if result['is_repetitive']:
                logger.info(f"Compression filter: detected repetition (ratio={result['compression_ratio']:.2f}, reason={result['reason']})")

                if result['reason'] == 'cross_segment_pattern':
                    # For cross-segment patterns like "hejsan hallå", try to trim head
                    return _trim_repetitive_head(words, state, result)
                else:
                    # For intra-segment repetition, use standard trimming with compression tie-breaking
                    if MAIN_DETECTOR_AVAILABLE:
                        return maybe_trim_repetition(words, state, ctx=ctx, compression_score=result['compression_ratio'])
                    else:
                        return words, "main_detector_unavailable", 0

        except Exception as e:
            logger.warning(f"Compression detection failed: {e}")

    # No compression issues detected - proceed with normal detection
    if MAIN_DETECTOR_AVAILABLE:
        return maybe_trim_repetition(words, state, ctx=ctx)
    else:
        return words, "no_filtering", 0


def _trim_repetitive_head(
    words: List[Word],
    state: DetectorState,
    compression_result: Dict
) -> Tuple[List[Word], Optional[str], int]:
    """
    Trim repetitive head patterns for cross-segment repetition.

    This handles cases like "hejsan hallå" appearing at the start of multiple segments.
    Strategy: Remove first 2-3 words if they match recent patterns.
    """
    if len(words) < 2:
        return words, "too_short_for_head_trim", 0

    segment_text = " ".join([w.text for w in words])
    head_text = " ".join([w.text for w in words[:3]])  # First 3 words

    # Check if head matches recent patterns in compression detector
    if hasattr(state, 'compression_detector') and state.compression_detector:
        stats = state.compression_detector.get_recent_compression_stats()

        # If we have recent context and this segment shows cross-segment repetition
        if stats.get('count', 0) > 0 and compression_result['reason'] == 'cross_segment_pattern':
            # Adaptive head trimming: try different trim lengths to find optimal cut point
            best_improvement = 0
            best_trim_count = 0
            best_trimmed_words = words

            # Try trimming 1-4 words from the beginning
            for trim_count in range(1, min(5, len(words))):
                if len(words) > trim_count:
                    trimmed_words = words[trim_count:]
                    trimmed_text = " ".join([w.text for w in trimmed_words])

                    # Check if trimmed version has better compression ratio
                    try:
                        detector = state.compression_detector
                        trimmed_ratio = detector.calculate_compression_ratio(trimmed_text)

                        # Calculate improvement (higher is better)
                        improvement = compression_result['compression_ratio'] - trimmed_ratio

                        # Keep the trim that gives the best improvement
                        # Require very significant improvement to avoid over-trimming normal content
                        if improvement > best_improvement and improvement > 0.20:
                            best_improvement = improvement
                            best_trim_count = trim_count
                            best_trimmed_words = trimmed_words

                    except Exception as e:
                        logger.debug(f"Error calculating trimmed compression ratio: {e}")
                        continue

            # Apply the best trim found
            if best_trim_count > 0:
                final_ratio = compression_result['compression_ratio'] - best_improvement
                logger.info(f"Adaptive head trim: removed {best_trim_count} words, compression improved {compression_result['compression_ratio']:.2f} -> {final_ratio:.2f}")
                return best_trimmed_words, f"adaptive_head_trim_{best_trim_count}_words", best_trim_count

    # If head trimming doesn't help, fall back to main detector
    if MAIN_DETECTOR_AVAILABLE:
        return maybe_trim_repetition(words, state, ctx=None)
    else:
        return words, "head_trim_failed", 0


def get_compression_stats(state: DetectorState) -> Dict:
    """Get compression detection stats for monitoring."""
    if hasattr(state, 'compression_detector') and state.compression_detector:
        return state.compression_detector.get_recent_compression_stats()
    return {"error": "compression_detector_unavailable"}


def reset_compression_context(state: DetectorState):
    """Reset compression detector context (for new sessions)."""
    if hasattr(state, 'compression_detector') and state.compression_detector:
        state.compression_detector.reset_context()


# Export key functions for easy integration
__all__ = [
    'compress_filter_segment',
    'get_compression_stats',
    'reset_compression_context',
    'COMPRESSION_AVAILABLE'
]