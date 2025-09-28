import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_machine import TranscriptionPipeline
from core.types import TranscriptChunk, AudioBlob, ASRResult, ASRSegment
from providers.base import ASRProvider

class _P(ASRProvider):
    name = "p"
    def __init__(self, text: str):
        self._t = text
    def transcribe_file(self, wav_path, language):
        if self._t == "":
            return ASRResult(ok=False, segments=[], raw_text="", provider=self.name, meta={}, error="asr_empty")
        return ASRResult(ok=True, segments=[ASRSegment(text=self._t, start_s=0, end_s=1)], raw_text=self._t, provider=self.name, meta={})

def test_non_empty_asr_always_appends(tmp_path):
    out = {}
    def wal_append(chunk: TranscriptChunk):
        out[chunk.seq] = chunk.text
        return f"mem://{chunk.seq}"
    pipe = TranscriptionPipeline(_P("hej"), None, wal_append)
    blob = AudioBlob(session_id="s", seq=1, captured_ts=0.0, wav_path="x.wav", duration_s=1.0)
    chunk = pipe.process_blob(blob, language="sv")
    assert chunk is not None
    assert out[1] == "hej"

def test_true_silence_drops(tmp_path):
    out = {}
    def wal_append(chunk: TranscriptChunk):
        out[chunk.seq] = chunk.text
        return f"mem://{chunk.seq}"
    pipe = TranscriptionPipeline(_P(""), None, wal_append)
    blob = AudioBlob(session_id="s", seq=1, captured_ts=0.0, wav_path="x.wav", duration_s=1.0)
    chunk = pipe.process_blob(blob, language="sv")
    assert chunk is None
    assert out == {}
