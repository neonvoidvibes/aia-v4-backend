from math import log2
from typing import List, Optional
import os

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
    dur = duration_s
    return {"rep_ratio": rep, "entropy": ent, "duration": dur, "window_len": len(window), "chars": len(joined)}

def drop_leading_initial_noise(tokens: List[str], duration_s: Optional[float]) -> int:
    """
    If the leading run qualifies as initial noise, return how many tokens to drop.
    We drop either the full leading repeat-run, or the entire short window if not a pure repeat.
    """
    if not tokens:
        return 0
    if is_initial_noise(tokens, duration_s):
        run = _leading_repeat_run(tokens)
        return max(run, len(tokens[:4]))  # drop ≥ repeat run; cap by short-window
    return 0