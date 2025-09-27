"""Post-ASR text selection and fallback logic."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

# Detector imports (required for repetition checks)
try:
    from utils.hallucination_detector import detect_repetition_in_text, DetectorState
except ImportError:
    # Detector fallbacks
    def detect_repetition_in_text(text: str, state: object) -> bool:
        return False
    class DetectorState:
        pass

# Metrics imports (optional)
try:
    from utils.hallucination_metrics_v2 import (
        metrics_collector, DropReason, FallbackReason, OutcomeType
    )
except ImportError:
    # Fallback if metrics not available
    class NullMetricsCollector:
        def track_post_asr_decision(self, *args, **kwargs): pass
        def track_drop(self, *args, **kwargs): pass
        def track_fallback(self, *args, **kwargs): pass
        def track_validator_outcome(self, *args, **kwargs): pass
        def track_pii_outcome(self, *args, **kwargs): pass
        def time_post_asr_decision(self, *args, **kwargs):
            class NullTimer:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return NullTimer()
    metrics_collector = NullMetricsCollector()
    # Enum fallbacks
    class DropReason:
        ASR_EMPTY = "asr_empty"
        FILTER_EMPTY = "filter_empty"
        PII_ERROR = "pii_error"
        PII_BLOCKED = "pii_blocked"
        REPETITION_BLOCKED = "repetition_blocked"
    class FallbackReason:
        PIPELINE_EMPTY = "pipeline_empty"
        MIN_SIZE = "min_size"
        PII_INVALID = "pii_invalid"
    class OutcomeType:
        PASS = "pass"
        FAIL = "fail"
        ERROR = "error"
        REDACT = "redact"
        REPETITION_BLOCKED = "repetition_blocked"


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
DROP_REPETITION_BLOCKED_FALLBACK = "REPETITION_BLOCKED_FALLBACK"

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
    provider: str = "unknown",
    language: str = "unknown",
    detector_state: Optional[DetectorState] = None,
) -> PostAsrDecision:
    """Select the transcript candidate to append, applying fallbacks when needed."""
    with metrics_collector.time_post_asr_decision(provider=provider, language=language):
        return _decide_transcript_candidate_impl(
            original_text, filtered_text, run_pii, validator,
            toggles=toggles, min_tokens=min_tokens, min_chars=min_chars,
            provider=provider, language=language, detector_state=detector_state
        )


def _decide_transcript_candidate_impl(
    original_text: str,
    filtered_text: str,
    run_pii: Callable[[str], str],
    validator: Callable[[str], bool],
    *,
    toggles: FeatureToggles,
    min_tokens: int = 2,
    min_chars: int = 6,
    provider: str = "unknown",
    language: str = "unknown",
    detector_state: Optional[DetectorState] = None,
) -> PostAsrDecision:
    """Implementation with metrics tracking."""
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
        decision = PostAsrDecision(
            final_text="",
            low_confidence=False,
            used_fallback=False,
            drop_reason=DROP_ASR_EMPTY,
            pii_pass=False,
            stats=stats,
            metadata={"candidate_source": "none"},
        )
        # Don't track ASR_EMPTY as false drop
        return decision

    candidate = filtered
    candidate_source = "filtered"
    fallback_reason: Optional[str] = None
    used_fallback = False
    low_confidence = False
    repetition_blocked_fallback = False

    if toggles.never_empty_contract and not candidate:
        # Check if original text contains repetition before falling back
        if detector_state and detect_repetition_in_text(original, detector_state):
            # Don't fall back to repetitive original text
            repetition_blocked_fallback = True
            # Track that we blocked a fallback due to repetition
            try:
                # Could add custom metric here: FALLBACK_BLOCKED_REPETITION.labels(reason="PIPELINE_EMPTY").inc()
                pass
            except:
                pass
        else:
            candidate = original
            candidate_source = "original"
            fallback_reason = FALLBACK_PIPELINE_EMPTY
            used_fallback = True
            low_confidence = True

    if toggles.min_size_guard and candidate:
        tokens = _count_tokens(candidate)
        if tokens < min_tokens or len(candidate) < min_chars:
            # Check if original text contains repetition before falling back
            if detector_state and detect_repetition_in_text(original, detector_state):
                # Don't fall back to repetitive original text - keep small filtered text
                # Track that we kept filtered text due to repetition blocking fallback
                try:
                    # Add metadata to track this decision for analytics
                    stats["kept_filtered_small_due_to_repetition_block"] = 1
                    # Could add custom metric here: FALLBACK_BLOCKED_REPETITION.labels(reason="MIN_SIZE").inc()
                except:
                    pass
                # Don't clear candidate - keep the small filtered text instead of dropping
            else:
                candidate = original
                candidate_source = "original"
                fallback_reason = FALLBACK_MIN_SIZE
                used_fallback = True
                low_confidence = True

    if not candidate:
        # No candidate even after fallback (likely because toggles disabled or repetition blocked)
        drop_reason = DROP_REPETITION_BLOCKED_FALLBACK if repetition_blocked_fallback else DROP_FILTER_EMPTY
        decision = PostAsrDecision(
            final_text="",
            low_confidence=False,
            used_fallback=used_fallback,
            drop_reason=drop_reason,
            pii_pass=False,
            stats=stats,
            metadata={
                "candidate_source": "none",
                "fallback_reason": fallback_reason,
                "repetition_blocked": repetition_blocked_fallback,
            },
        )
        # Track drop
        metrics_collector.track_post_asr_decision("dropped", provider=provider, language=language)
        if repetition_blocked_fallback:
            metrics_collector.track_drop(DropReason.REPETITION_BLOCKED, provider=provider, language=language)
        else:
            metrics_collector.track_drop(DropReason.FILTER_EMPTY, provider=provider, language=language)
        return decision

    def _run_pii_candidate(text: str) -> Optional[str]:
        try:
            result = run_pii(text)
            if result != text:
                metrics_collector.track_pii_outcome(OutcomeType.REDACT, provider=provider, language=language)
            else:
                metrics_collector.track_pii_outcome(OutcomeType.PASS, provider=provider, language=language)
            return result
        except Exception:
            metrics_collector.track_pii_outcome(OutcomeType.ERROR, provider=provider, language=language)
            return None

    pii_result = _run_pii_candidate(candidate)
    if pii_result is None:
        decision = PostAsrDecision(
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
        # Track drop
        metrics_collector.track_post_asr_decision("dropped", provider=provider, language=language)
        metrics_collector.track_drop(DropReason.PII_ERROR, provider=provider, language=language)
        return decision

    if validator(pii_result):
        stats["pii_pass"] = 1
        metadata = {
            "candidate_source": candidate_source,
            "fallback_reason": fallback_reason,
            "final_source": candidate_source,
        }
        decision = PostAsrDecision(
            final_text=pii_result,
            low_confidence=low_confidence,
            used_fallback=used_fallback,
            drop_reason=DROP_NONE,
            pii_pass=True,
            stats=stats,
            metadata=metadata,
        )
        # Track validator pass
        metrics_collector.track_validator_outcome(OutcomeType.PASS, provider=provider, language=language)
        # Track successful processing or fallback
        if used_fallback:
            metrics_collector.track_post_asr_decision("fallback", provider=provider, language=language)
            # Map fallback reason to enum
            if fallback_reason == FALLBACK_PIPELINE_EMPTY:
                metrics_collector.track_fallback(FallbackReason.PIPELINE_EMPTY, provider=provider, language=language)
            elif fallback_reason == FALLBACK_MIN_SIZE:
                metrics_collector.track_fallback(FallbackReason.MIN_SIZE, provider=provider, language=language)
        else:
            metrics_collector.track_post_asr_decision("kept", provider=provider, language=language)
        return decision

    # PII rejected the filtered candidate; attempt fallback to original if allowed
    if toggles.never_empty_contract and candidate_source != "original" and original:
        # Check if original text contains repetition before falling back
        if detector_state and detect_repetition_in_text(original, detector_state):
            # Don't fall back to repetitive original text - skip PII fallback
            # Track that we blocked a PII fallback due to repetition
            try:
                metrics_collector.track_pii_outcome(OutcomeType.REPETITION_BLOCKED, provider=provider, language=language)
                # Could add custom metric here: FALLBACK_BLOCKED_REPETITION.labels(reason="PII_INVALID").inc()
            except:
                pass
        else:
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
    decision = PostAsrDecision(
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
    # Track validator fail and drop
    metrics_collector.track_validator_outcome(OutcomeType.FAIL, provider=provider, language=language)
    metrics_collector.track_post_asr_decision("dropped", provider=provider, language=language)
    metrics_collector.track_drop(DropReason.PII_BLOCKED, provider=provider, language=language)
    return decision
