from __future__ import annotations
import os, requests, logging
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

class DeepgramProvider(TranscriptionProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "nova-3", smart_format: bool = True, punctuate: bool = True):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            logger.error("DEEPGRAM_API_KEY is not set")
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self.model = model
        self.smart_format = smart_format
        self.punctuate = punctuate
        self._endpoint = "https://api.deepgram.com/v1/listen"
        logger.info(f"DeepgramProvider initialized with model={model} (multilingual capable), api_key={'*' * 10}{'...' if len(self.api_key) > 10 else ''}")

    def _get_deepgram_vad_mode(self, vad_aggressiveness: Optional[int]) -> Optional[int]:
        """
        Map frontend VAD aggressiveness (1=Quiet, 2=Mid, 3=Noisy) to Deepgram VAD mode.

        Frontend mapping:
        - 1 = Quiet environment (less aggressive VAD needed)
        - 2 = Mid environment (balanced VAD)
        - 3 = Noisy environment (more aggressive VAD needed)

        Deepgram vad_mode:
        - 0 = Least aggressive (more inclusive)
        - 1 = Default (balanced)
        - 2 = More aggressive
        - 3 = Most aggressive (highly restrictive)
        """
        if vad_aggressiveness is None:
            return 1  # Default to Quiet for Deepgram (reduces hallucination)

        # Map frontend levels to Deepgram VAD modes
        # Keep mapping in [1..3] range as recommended
        mapping = {
            1: 1,  # Quiet -> Default balanced mode
            2: 2,  # Mid -> More aggressive
            3: 3   # Noisy -> Most aggressive
        }
        return mapping.get(vad_aggressiveness, 1)  # Default to balanced (1) if invalid

    def _request(self, path: str, language: Optional[str], vad_aggressiveness: Optional[int] = None, timeout_s: float = None, max_retries: int = None) -> Dict[str, Any]:
        import httpx, time, math

        # Get timeout and retry settings from environment or use defaults
        timeout_s = timeout_s if timeout_s is not None else float(os.getenv("DEEPGRAM_TIMEOUT_MS", "6000")) / 1000.0
        max_retries = max_retries if max_retries is not None else int(os.getenv("DEEPGRAM_MAX_RETRIES", "2"))

        # Deepgram parameters - enable word timestamps and smart formatting
        params = {
            "model": self.model,
            "smart_format": str(self.smart_format).lower(),
            "punctuate": str(self.punctuate).lower(),
            "words": "true",  # Enable word-level timestamps
            "timestamps": "true"  # Enable timestamps
        }
        # Handle language parameter for nova-3 multilingual support
        if language and language != "any":
            params["language"] = language
        else:
            # For nova-3, enable multilingual detection when language is "any"
            params["detect_language"] = "true"

        # Add VAD mode if configured
        vad_mode = self._get_deepgram_vad_mode(vad_aggressiveness)
        if vad_mode is not None:
            params["vad_mode"] = str(vad_mode)

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav"  # Specify content type
        }

        logger.info(f"Deepgram request: file={path}, params={params}, timeout={timeout_s}s, max_retries={max_retries}")

        def _do():
            with open(path, "rb") as f:
                with httpx.Client(timeout=httpx.Timeout(timeout_s)) as cli:
                    r = cli.post(self._endpoint, headers=headers, params=params, data=f)

                    logger.info(f"Deepgram response: status={r.status_code}")

                    if r.status_code >= 300:
                        logger.error(f"Deepgram HTTP {r.status_code}: {r.text[:500]}")
                        raise RuntimeError(f"Deepgram HTTP {r.status_code}: {r.text[:500]}")

                    response_data = r.json()
                    logger.info(f"Deepgram response data keys: {list(response_data.keys())}")
                    return response_data

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return _do()
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    # backoff with jitter
                    sleep_time = min(0.2 * (2 ** attempt), 1.2) + (0.05 * math.tanh(attempt))
                    logger.warning(f"Deepgram attempt {attempt + 1} failed: {e}. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Deepgram request failed after {max_retries + 1} attempts: {e}")

        raise last_exc

    def _extract(self, data: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
        try:
            alt0 = data["results"]["channels"][0]["alternatives"][0]
            return (alt0.get("transcript","").strip(), alt0.get("words") or [])
        except Exception:
            return ("", [])

    def _words_to_segments(self, words: List[Dict[str, Any]], gap_s: float = 0.6) -> List[Segment]:
        segs: List[Segment] = []
        if not words: return segs
        cur = {"start": float(words[0]["start"]), "end": float(words[0]["end"]), "text": [words[0]["word"]]}
        for prev, w in zip(words, words[1:]):
            w_start, w_end = float(w["start"]), float(w["end"])
            if (w_start - float(prev["end"])) > gap_s:
                segs.append({"start": cur["start"], "end": cur["end"], "text": " ".join(cur["text"]).strip(),
                             "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0})
                cur = {"start": w_start, "end": w_end, "text": [w["word"]]}
            else:
                cur["end"] = w_end
                cur["text"].append(w["word"])
        segs.append({"start": cur["start"], "end": cur["end"], "text": " ".join(cur["text"]).strip(),
                     "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0})
        return segs

    def transcribe_file(self, path: str, language: Optional[str] = None, prompt: Optional[str] = None, vad_aggressiveness: Optional[int] = None) -> Optional[TranscriptionResult]:
        try:
            vad_info = f", VAD={self._get_deepgram_vad_mode(vad_aggressiveness)}" if vad_aggressiveness else ""
            logger.info(f"Deepgram transcribe_file called: path={path}, language={language}{vad_info}")
            data = self._request(path, language, vad_aggressiveness)
            text, words = self._extract(data)
            segments = self._words_to_segments(words)

            logger.info(f"Deepgram transcription result: text_length={len(text)}, segments_count={len(segments)}")
            if text:
                logger.info(f"Deepgram transcript preview: {text[:100]}...")

            # Apply initial noise detection (phrase repetition filtering)
            if text and words:
                tokens = [w.get("word", w.get("text", "")).strip() for w in words if w.get("word") or w.get("text")]

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
                            f"DEEPGRAM_INITIAL_NOISE_DROP path={os.path.basename(path)} drop_n={drop_count} "
                            f"rep={signals.get('rep_ratio', 0):.3f} ent={signals.get('entropy', 0):.3f} "
                            f"phrase_rep={signals.get('phrase_repetition', 0):.3f} dur={duration_s} "
                            f"win={signals.get('window_len', 0)} chars={signals.get('chars', 0)} "
                            f"original_text='{text[:50]}...'"
                        )

                        # Remove the leading noise tokens from words and update text/segments
                        filtered_words = words[drop_count:] if drop_count < len(words) else []

                        if filtered_words:
                            # Rebuild text from filtered words
                            filtered_text = " ".join(w.get("word", w.get("text", "")).strip()
                                                   for w in filtered_words
                                                   if w.get("word") or w.get("text")).strip()

                            # Rebuild segments from filtered words
                            filtered_segments = self._words_to_segments(filtered_words)

                            logger.info(f"Deepgram noise filtering: removed {drop_count} tokens, "
                                       f"text: '{text[:50]}...' -> '{filtered_text[:50]}...'")

                            text = filtered_text
                            words = filtered_words
                            segments = filtered_segments
                        else:
                            # All tokens were noise - return empty result
                            logger.info(f"Deepgram noise filtering: all {drop_count} tokens were noise, returning empty result")
                            text = ""
                            words = []
                            segments = []

# Normalize to service Word schema in upstream caller; ensure 'confidence' key exists.
            words_normalized = [
                {"text": w.get("word") or w.get("text",""), "start": w["start"], "end": w["end"], "confidence": w.get("confidence", 1.0)}
                for w in words
            ]
            result = {"text": text, "segments": segments, "words": words_normalized}
            return result
        except Exception as e:
            logger.error(f"Deepgram transcribe_file failed: {e}")
            return None