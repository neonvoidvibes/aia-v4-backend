"""Batch transcription utilities that reuse the realtime pipeline for S3 jobs."""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Callable, List, Optional

from core.state_machine import TranscriptionPipeline
from core.types import AudioBlob, TranscriptChunk
from core.wav_norm import to_mono16k_pcm
from providers.deepgram_provider import DeepgramProvider
from providers.whisper_provider import WhisperProvider

LoggerLike = logging.Logger
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class SegmentSpec:
    """Metadata for a normalized wav slice."""
    path: str
    start_s: float
    duration_s: float
    index: int


class BatchTranscriptionError(RuntimeError):
    pass


def probe_media_duration(path: str) -> Optional[float]:
    """Return duration (seconds) using ffprobe when available."""
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


class BatchTranscriber:
    """Normalize, slice, and transcribe long-form audio via the pipeline."""

    def __init__(
        self,
        *,
        dg_client,
        whisper_client,
        provider_name: str,
        chunk_seconds: int,
        max_parallel: int = 1,
        logger: Optional[LoggerLike] = None,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")
        self._dg_client = dg_client
        self._whisper_client = whisper_client
        self._provider_name = (provider_name or "deepgram").lower()
        self._chunk_seconds = chunk_seconds
        self._max_parallel = max(1, max_parallel)
        self._logger = logger or logging.getLogger(__name__)

    def transcribe(
        self,
        *,
        source_path: str,
        language: Optional[str],
        session_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        start_time = time.time()
        with tempfile.TemporaryDirectory(prefix="batch_transcribe_") as workdir:
            normalized_path, duration_s = self._normalize(source_path, workdir)
            segments = self._segment(normalized_path, duration_s, workdir)

            if not segments:
                return {
                    "text": "",
                    "segments": [],
                    "duration": duration_s,
                    "chunks": 0,
                    "processing_ms": 0,
                    "segments_count": 0,
                }

            pipeline = self._build_pipeline()
            aggregated_text: List[str] = []
            aggregated_segments: List[dict] = []
            processing_times: List[float] = []

            total = len(segments)
            capture_epoch = time.time()

            for idx, spec in enumerate(segments):
                if progress_callback:
                    progress_callback(idx, total, f"Transcribing segment {idx + 1} / {total}")
                segment_start = time.time()
                chunk = self._process_segment(
                    pipeline=pipeline,
                    session_id=session_id,
                    spec=spec,
                    language=language,
                    captured_ts=capture_epoch + spec.start_s,
                )
                processing_times.append((time.time() - segment_start) * 1000.0)
                if not chunk:
                    continue
                aggregated_text.append(chunk.text)

                meta = chunk.meta or {}
                segment_payload = meta.get("segments") or []
                if isinstance(segment_payload, int):
                    segment_payload = []
                for local_idx, seg in enumerate(segment_payload):
                    start = float(seg.get("start", 0.0)) + spec.start_s
                    end = float(seg.get("end", start)) + spec.start_s
                    text = seg.get("text", "").strip()
                    if not text:
                        continue
                    aggregated_segments.append({
                        "id": len(aggregated_segments),
                        "seek": 0,
                        "start": start,
                        "end": end,
                        "text": text,
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                        "segment_index": spec.index,
                        "segment_local_index": local_idx,
                        "provider": chunk.provider,
                    })
                if progress_callback:
                    progress_callback(idx + 1, total, f"Completed segment {idx + 1} / {total}")

            total_ms = (time.time() - start_time) * 1000.0
            if progress_callback:
                progress_callback(total, total, "Transcription complete")

            if aggregated_text:
                full_text = "\n".join(t.strip() for t in aggregated_text if t.strip())
            else:
                full_text = ""

            mean_ms = sum(processing_times) / len(processing_times) if processing_times else 0.0
            self._logger.info(
                "Batch transcription complete: session=%s provider=%s segments=%s total_ms=%.1f mean_segment_ms=%.1f",
                session_id,
                self._provider_name,
                len(aggregated_segments),
                total_ms,
                mean_ms,
            )

            return {
                "text": full_text,
                "segments": aggregated_segments,
                "duration": duration_s,
                "chunks": total,
                "segments_count": len(aggregated_segments),
                "processing_ms": total_ms,
                "mean_segment_ms": mean_ms,
            }

    def _normalize(self, src_path: str, workdir: str) -> tuple[str, float]:
        dst_path = os.path.join(workdir, "normalized.wav")
        self._logger.debug("Normalizing %s -> %s", src_path, dst_path)
        to_mono16k_pcm(src_path, dst_path)
        with contextlib.closing(wave.open(dst_path, "rb")) as wf:
            frames = wf.getnframes()
            sample_rate = wf.getframerate()
            duration_s = frames / float(sample_rate or 1)
        return dst_path, duration_s

    def _segment(self, wav_path: str, duration_s: float, workdir: str) -> List[SegmentSpec]:
        segments: List[SegmentSpec] = []
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            channels = wf.getnchannels()
            if sample_rate <= 0:
                raise BatchTranscriptionError("Invalid sample rate in normalized audio")

            chunk_frames = int(self._chunk_seconds * sample_rate)
            total_frames = wf.getnframes()
            if chunk_frames <= 0:
                chunk_frames = total_frames

            index = 0
            start_frame = 0
            while start_frame < total_frames:
                frames_to_read = min(chunk_frames, total_frames - start_frame)
                wf.setpos(start_frame)
                raw = wf.readframes(frames_to_read)
                if not raw:
                    break
                slice_path = os.path.join(workdir, f"segment_{index:04d}.wav")
                with contextlib.closing(wave.open(slice_path, "wb")) as out:
                    out.setnchannels(channels)
                    out.setsampwidth(sampwidth)
                    out.setframerate(sample_rate)
                    out.writeframes(raw)

                start_s = start_frame / float(sample_rate)
                duration_chunk = frames_to_read / float(sample_rate)
                segments.append(SegmentSpec(path=slice_path, start_s=start_s, duration_s=duration_chunk, index=index))
                index += 1
                start_frame += frames_to_read

        if not segments and duration_s > 0:
            segments.append(SegmentSpec(path=wav_path, start_s=0.0, duration_s=duration_s, index=0))
        return segments

    def _build_pipeline(self) -> TranscriptionPipeline:
        primary, fallback = self._resolve_providers()

        def _wal_append(chunk: TranscriptChunk) -> str:
            return f"memory://{chunk.session_id}/{chunk.seq}"

        return TranscriptionPipeline(provider_primary=primary, provider_fallback=fallback, wal_append=_wal_append)

    def _resolve_providers(self):
        name = self._provider_name
        dg_provider = DeepgramProvider(self._dg_client)
        whisper_provider = WhisperProvider(self._whisper_client)

        if name == "whisper" or name == "openai":
            primary = whisper_provider
            fallback = dg_provider
        else:
            primary = dg_provider
            fallback = whisper_provider
        return primary, fallback

    def _process_segment(
        self,
        *,
        pipeline: TranscriptionPipeline,
        session_id: str,
        spec: SegmentSpec,
        language: Optional[str],
        captured_ts: float,
    ) -> Optional[TranscriptChunk]:
        blob = AudioBlob(
            session_id=session_id,
            seq=spec.index,
            captured_ts=captured_ts,
            wav_path=spec.path,
            duration_s=spec.duration_s,
        )
        return pipeline.process_blob(blob, language=language)
