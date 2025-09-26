"""Post-ASR text selection and fallback logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


def _count_tokens(text: str) -> int:
    return len(text.split()) if text else 0


@dataclass
class FeatureToggles:
    """Runtime feature toggles controlling fallback behaviour."""
    never_empty_contract: bool = True
    min_size_guard: bool = True


@dataclass
class PostAsrDecision:
    final_text: str
    low_confidence: bool
    used_fallback: bool
    drop_reason: str
    pii_pass: bool
    stats: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


DROP_NONE = "NONE"
DROP_ASR_EMPTY = "ASR_EMPTY"
DROP_FILTER_EMPTY = "FILTER_EMPTY"
DROP_FILTER_INVALID = "FILTERED_INVALID"
DROP_PII_ORIGINAL = "PII_BLOCKED_ORIGINAL"
DROP_PII_ERROR = "PII_ERROR"

FALLBACK_PIPELINE_EMPTY = "PIPELINE_EMPTY_BUT_FALLBACK_USED"
FALLBACK_MIN_SIZE = "MIN_SIZE_FALLBACK"
FALLBACK_PII_INVALID = "PII_INVALID_FALLBACK"


def decide_transcript_candidate(
    original_text: str,
    filtered_text: str,
    run_pii: Callable[[str], str],
    validator: Callable[[str], bool],
    *,
    toggles: FeatureToggles,
    min_tokens: int = 2,
    min_chars: int = 6,
) -> PostAsrDecision:
    """Select the transcript candidate to append, applying fallbacks when needed."""
    original = (original_text or "").strip()
    filtered = (filtered_text or "").strip()

    stats: Dict[str, int] = {
        "pre_len_chars": len(original),
        "pre_len_tokens": _count_tokens(original),
        "post_trim_chars": len(filtered),
        "post_trim_tokens": _count_tokens(filtered),
        "post_dedup_tokens": _count_tokens(filtered),
    }

    # Early exit when ASR produced nothing
    if not original:
        return PostAsrDecision(
            final_text="",
            low_confidence=False,
            used_fallback=False,
            drop_reason=DROP_ASR_EMPTY,
            pii_pass=False,
            stats=stats,
            metadata={"candidate_source": "none"},
        )

    candidate = filtered
    candidate_source = "filtered"
    fallback_reason: Optional[str] = None
    used_fallback = False
    low_confidence = False

    if toggles.never_empty_contract and not candidate:
        candidate = original
        candidate_source = "original"
        fallback_reason = FALLBACK_PIPELINE_EMPTY
        used_fallback = True
        low_confidence = True

    if toggles.min_size_guard and candidate:
        tokens = _count_tokens(candidate)
        if tokens < min_tokens or len(candidate) < min_chars:
            candidate = original
            candidate_source = "original"
            fallback_reason = FALLBACK_MIN_SIZE
            used_fallback = True
            low_confidence = True

    if not candidate:
        # No candidate even after fallback (likely because toggles disabled)
        return PostAsrDecision(
            final_text="",
            low_confidence=False,
            used_fallback=used_fallback,
            drop_reason=DROP_FILTER_EMPTY,
            pii_pass=False,
            stats=stats,
            metadata={
                "candidate_source": "none",
                "fallback_reason": fallback_reason,
            },
        )

    def _run_pii_candidate(text: str) -> Optional[str]:
        try:
            return run_pii(text)
        except Exception:
            return None

    pii_result = _run_pii_candidate(candidate)
    if pii_result is None:
        return PostAsrDecision(
            final_text="",
            low_confidence=low_confidence,
            used_fallback=used_fallback,
            drop_reason=DROP_PII_ERROR,
            pii_pass=False,
            stats=stats,
            metadata={
                "candidate_source": candidate_source,
                "fallback_reason": fallback_reason,
            },
        )

    if validator(pii_result):
        stats["pii_pass"] = 1
        metadata = {
            "candidate_source": candidate_source,
            "fallback_reason": fallback_reason,
            "final_source": candidate_source,
        }
        return PostAsrDecision(
            final_text=pii_result,
            low_confidence=low_confidence,
            used_fallback=used_fallback,
            drop_reason=DROP_NONE,
            pii_pass=True,
            stats=stats,
            metadata=metadata,
        )

    # PII rejected the filtered candidate; attempt fallback to original if allowed
    if toggles.never_empty_contract and candidate_source != "original" and original:
        fallback_candidate = original
        fallback_reason = FALLBACK_PII_INVALID
        used_fallback = True
        low_confidence = True

        pii_result = _run_pii_candidate(fallback_candidate)
        if pii_result and validator(pii_result):
            stats["pii_pass"] = 1
            stats["pii_fallback"] = 1
            return PostAsrDecision(
                final_text=pii_result,
                low_confidence=low_confidence,
                used_fallback=used_fallback,
                drop_reason=DROP_NONE,
                pii_pass=True,
                stats=stats,
                metadata={
                    "candidate_source": "filtered",
                    "fallback_reason": fallback_reason,
                    "final_source": "original",
                },
            )

    # Final failure path: record PII rejection
    stats["pii_pass"] = 0
    return PostAsrDecision(
        final_text="",
        low_confidence=low_confidence,
        used_fallback=used_fallback,
        drop_reason=DROP_PII_ORIGINAL,
        pii_pass=False,
        stats=stats,
        metadata={
            "candidate_source": candidate_source,
            "fallback_reason": fallback_reason,
            "final_source": "none",
        },
    )
