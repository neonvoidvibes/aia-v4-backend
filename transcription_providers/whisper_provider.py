from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional
from .base import TranscriptionProvider, TranscriptionResult, Segment

logger = logging.getLogger(__name__)

# Import noise detection for phrase repetition filtering
try:
    from utils.text_noise import drop_leading_initial_noise, noise_signals
except ImportError:
    logger.warning("text_noise module not available - initial noise detection disabled")
    def drop_leading_initial_noise(tokens, duration): return 0
    def noise_signals(tokens, duration): return {}

_USE_INTERNAL_HELPER = False
try:
    from services.transcription_helpers import transcribe_with_openai as _openai_helper  # pragma: no cover
    _USE_INTERNAL_HELPER = True
except Exception:
    pass

class WhisperProvider(TranscriptionProvider):
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "whisper-1"):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        if not _USE_INTERNAL_HELPER:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key)

    def _normalize_segments(self, segments: List[Any]) -> List[Segment]:
        out: List[Segment] = []
        for s in segments or []:
            # Handle both dict objects (internal helper) and Pydantic objects (OpenAI API)
            if hasattr(s, 'get'):  # Dictionary-like object
                start = float(s.get("start", s.get("start_time", 0.0)))
                end = float(s.get("end", s.get("end_time", start)))
                text = (s.get("text") or s.get("transcript") or "").strip()
                avg_logprob = float(s.get("avg_logprob", 0.0))
                compression_ratio = float(s.get("compression_ratio", 0.0))
                no_speech_prob = float(s.get("no_speech_prob", 0.0))
            else:  # Pydantic object (OpenAI API response)
                start = float(getattr(s, "start", getattr(s, "start_time", 0.0)))
                end = float(getattr(s, "end", getattr(s, "end_time", start)))
                text = (getattr(s, "text", None) or getattr(s, "transcript", None) or "").strip()
                avg_logprob = float(getattr(s, "avg_logprob", 0.0))
                compression_ratio = float(getattr(s, "compression_ratio", 0.0))
                no_speech_prob = float(getattr(s, "no_speech_prob", 0.0))

            out.append({
                "start": start, "end": end, "text": text,
                "avg_logprob": avg_logprob,
                "compression_ratio": compression_ratio,
                "no_speech_prob": no_speech_prob,
            })
        return out

    def _align_words_from_segments(self, segments: List[Segment]) -> List[Dict[str, Any]]:
        """Provide 'words' with start/end/conf so the detector can run consistently."""
        words = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if not text:
                continue

            # Simple word splitting with approximate timing
            words_in_seg = text.split()
            if not words_in_seg:
                continue

            seg_duration = seg.get('end', 0) - seg.get('start', 0)
            word_duration = seg_duration / len(words_in_seg) if len(words_in_seg) > 0 else 0

            for i, word in enumerate(words_in_seg):
                word_start = seg.get('start', 0) + (i * word_duration)
                word_end = word_start + word_duration
                words.append({
                    "text": word,
                    "start": word_start,
                    "end": word_end,
                    "confidence": 1.0  # Whisper doesn't provide word-level confidence
                })

        return words

    def transcribe_file(self, path: str, language: Optional[str] = None, prompt: Optional[str] = None, vad_aggressiveness: Optional[int] = None) -> Optional[TranscriptionResult]:
        # Handle language parameter: OpenAI Whisper uses None for auto-detect, not 'any'
        openai_language = None if not language or language == "any" else language

        if _USE_INTERNAL_HELPER:
            data = _openai_helper(path=path, language=openai_language, prompt=prompt)
            if not data: return None
            text = (data.get("text") or "").strip()
            segments = self._normalize_segments(data.get("segments") or [])
            words = self._align_words_from_segments(segments)
        else:
            with open(path, "rb") as f:
                resp = self._client.audio.transcriptions.create(  # type: ignore
                    file=f, model=self.model, language=openai_language, prompt=prompt, response_format="verbose_json", temperature=0
                )
            text = (getattr(resp, "text", None) or getattr(resp, "get", lambda *_: None)("text") or "").strip()  # type: ignore
            segments_raw = getattr(resp, "segments", None) or (isinstance(resp, dict) and resp.get("segments") or [])
            segments = self._normalize_segments(segments_raw or [])
            words = self._align_words_from_segments(segments)

        # Apply initial noise detection (phrase repetition filtering)
        if text and words:
            tokens = [w.get("text", "").strip() for w in words if w.get("text")]

            if tokens:
                # Calculate duration from first to last word
                duration_s = None
                try:
                    if len(words) > 0:
                        first_start = float(words[0].get("start", 0))
                        last_end = float(words[-1].get("end", 0))
                        duration_s = last_end - first_start
                except (ValueError, TypeError):
                    duration_s = None

                # Check for initial noise/phrase repetition
                drop_count = drop_leading_initial_noise(tokens, duration_s)

                if drop_count > 0:
                    # Log the detection with detailed signals
                    signals = noise_signals(tokens, duration_s)
                    logger.info(
                        f"WHISPER_INITIAL_NOISE_DROP path={os.path.basename(path)} drop_n={drop_count} "
                        f"rep={signals.get('rep_ratio', 0):.3f} ent={signals.get('entropy', 0):.3f} "
                        f"phrase_rep={signals.get('phrase_repetition', 0):.3f} dur={duration_s} "
                        f"win={signals.get('window_len', 0)} chars={signals.get('chars', 0)} "
                        f"original_text='{text[:50]}...'"
                    )

                    # Remove the leading noise tokens from words and update text/segments
                    filtered_words = words[drop_count:] if drop_count < len(words) else []

                    if filtered_words:
                        # Rebuild text from filtered words
                        filtered_text = " ".join(w.get("text", "").strip()
                                               for w in filtered_words
                                               if w.get("text")).strip()

                        # Rebuild segments from filtered words
                        filtered_segments = self._normalize_segments(segments[drop_count:] if drop_count < len(segments) else [])

                        logger.info(f"Whisper noise filtering: removed {drop_count} tokens, "
                                   f"text: '{text[:50]}...' -> '{filtered_text[:50]}...'")

                        text = filtered_text
                        words = filtered_words
                        segments = filtered_segments
                    else:
                        # All tokens were noise - return empty result
                        logger.info(f"Whisper noise filtering: all {drop_count} tokens were noise, returning empty result")
                        text = ""
                        words = []
                        segments = []

        return {"text": text, "segments": segments, "words": words}