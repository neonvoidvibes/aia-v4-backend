from math import log2
from typing import List, Optional, Set
import os
import re

def _shannon_entropy_chars(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c/n) * log2(c/n) for c in counts.values())

def _repetition_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    # fraction of tokens equal to the most frequent token
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return max(counts.values()) / len(tokens)

def _leading_repeat_run(tokens: List[str]) -> int:
    if not tokens:
        return 0
    first = tokens[0]
    i = 0
    while i < len(tokens) and tokens[i] == first:
        i += 1
    return i

def is_initial_noise(
    tokens: List[str],
    duration_s: Optional[float],
    text_joined: Optional[str] = None,
    *,
    max_tokens_window: int = int(os.getenv("NOISE_MAX_TOKENS_WINDOW", "4")),
    max_chars_window: int = int(os.getenv("NOISE_MAX_CHARS_WINDOW", "24")),
    max_duration_s: float = float(os.getenv("NOISE_MAX_DURATION_S", "1.6")),
    min_entropy_bits: float = float(os.getenv("NOISE_MIN_ENTROPY_BITS", "2.3")),
    min_repetition_ratio: float = float(os.getenv("NOISE_MIN_REP_RATIO", "0.6")),
) -> bool:
    """
    Language-agnostic heuristic for initial 'chant/noise' segments:
      - very short token window (<=4) OR short character span (<=24)
      - short duration (<=1.6s) if provided
      - low character entropy OR high repetition ratio
    All checks are local to the *leading* window of tokens.
    """
    if not tokens:
        return False
    window = tokens[:max_tokens_window]
    joined = "".join(window) if text_joined is None else text_joined
    chars_ok = len(joined) <= max_chars_window
    tokens_ok = len(window) <= max_tokens_window
    dur_ok = True if duration_s is None else (duration_s <= max_duration_s)

    rep = _repetition_ratio(window)
    ent = _shannon_entropy_chars(joined)
    signal_low_info = (rep >= min_repetition_ratio) or (ent <= min_entropy_bits)

    return (tokens_ok or chars_ok) and dur_ok and signal_low_info

def noise_signals(tokens: List[str], duration_s: Optional[float]):
    window = tokens[:max(1, int(os.getenv("NOISE_MAX_TOKENS_WINDOW", "4")))]
    joined = "".join(window)
    rep = _repetition_ratio(window)
    ent = _shannon_entropy_chars(joined)
    phrase_rep = _detect_phrase_repetition(tokens)
    dur = duration_s
    return {
        "rep_ratio": rep,
        "entropy": ent,
        "phrase_repetition": phrase_rep,
        "duration": dur,
        "window_len": len(window),
        "chars": len(joined)
    }

def _normalize_for_comparison(text: str) -> str:
    """Normalize text for phrase comparison - removes punctuation, extra spaces, lowercases."""
    # Remove punctuation and normalize whitespace
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def _detect_phrase_repetition(tokens: List[str], max_window: int = 12) -> float:
    """
    Industry-standard approach: detect repeated phrases using sliding n-gram windows.
    Returns the maximum repetition ratio found across different n-gram sizes.
    """
    if len(tokens) < 3:
        return 0.0

    max_repetition = 0.0
    full_text = " ".join(tokens)
    normalized_full = _normalize_for_comparison(full_text)

    # Try different n-gram sizes (2-word, 3-word, etc. phrases)
    for n in range(2, min(len(tokens)//2 + 1, 7)):  # up to 6-word phrases
        ngrams = []
        normalized_tokens = normalized_full.split()

        if len(normalized_tokens) < n:
            continue

        # Generate n-grams
        for i in range(len(normalized_tokens) - n + 1):
            ngram = " ".join(normalized_tokens[i:i+n])
            ngrams.append(ngram)

        if not ngrams:
            continue

        # Count occurrences
        counts = {}
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1

        # Calculate max repetition ratio for this n-gram size
        if counts:
            max_count = max(counts.values())
            repetition_ratio = max_count / len(ngrams)
            max_repetition = max(max_repetition, repetition_ratio)

    return max_repetition

def _detect_opening_phrase_repetition(tokens: List[str]) -> tuple[bool, int]:
    """
    Specialized detection for 'initial chant' patterns where the same phrase starts multiple segments.
    This handles cases like: "Okej då provar vi igen. [content]. Okej då provar vi igen."
    Returns (is_repetitive, tokens_to_drop)
    """
    if len(tokens) < 8:  # Need reasonable content to detect patterns
        return False, 0

    # Look for a phrase that appears at the beginning AND later in the sequence
    # Try different phrase lengths at the start
    for phrase_len in range(3, min(8, len(tokens)//2)):
        start_phrase = tokens[:phrase_len]
        start_phrase_normalized = _normalize_for_comparison(" ".join(start_phrase))

        # Check if this starting phrase appears again later
        for start_pos in range(phrase_len + 2, len(tokens) - phrase_len + 1):  # Skip some tokens, then check
            candidate_phrase = tokens[start_pos:start_pos + phrase_len]
            candidate_normalized = _normalize_for_comparison(" ".join(candidate_phrase))

            # Allow for slight variations in phrase matching
            similarity = _phrase_similarity(start_phrase_normalized, candidate_normalized)
            if similarity >= 0.8:  # High similarity threshold
                # Found repetitive opening phrase - drop the first occurrence
                return True, phrase_len

    return False, 0

def _phrase_similarity(phrase1: str, phrase2: str) -> float:
    """Calculate similarity between two normalized phrases."""
    if not phrase1 or not phrase2:
        return 0.0

    if phrase1 == phrase2:
        return 1.0

    # Simple token-based similarity
    tokens1 = set(phrase1.split())
    tokens2 = set(phrase2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0

def drop_leading_initial_noise(tokens: List[str], duration_s: Optional[float]) -> int:
    """
    Industry-standard initial noise detection combining multiple approaches.
    """
    if not tokens:
        return 0

    # Legacy window-based approach (for simple cases like "ah ah ah")
    if is_initial_noise(tokens, duration_s):
        run = _leading_repeat_run(tokens)
        return max(run, len(tokens[:4]))

    # Specialized opening phrase repetition detection (for "initial chants")
    is_opening_repetitive, opening_drop_count = _detect_opening_phrase_repetition(tokens)
    if is_opening_repetitive:
        return opening_drop_count

    # General phrase-repetition detection for other complex cases
    phrase_repetition = _detect_phrase_repetition(tokens)
    phrase_threshold = float(os.getenv("NOISE_PHRASE_REP_THRESHOLD", "0.25"))
    max_phrase_window = int(os.getenv("NOISE_MAX_PHRASE_WINDOW", "10"))

    # If significant phrase repetition detected, drop the leading instances
    if phrase_repetition >= phrase_threshold:
        window_to_check = tokens[:max_phrase_window]
        return len(window_to_check)

    return 0