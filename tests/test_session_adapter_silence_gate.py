import json
import math
import os
import sys
import wave
from array import array
import shutil

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.session_adapter as sa
from utils.silence_gate import SilenceGateResult, SilenceGateConfig


class _FakeS3:
    def __init__(self):
        self.put_calls = []

    def put_object(self, **kwargs):  # pragma: no cover - trivial
        self.put_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_paginator(self, name):  # pragma: no cover - unused in tests
        raise NotImplementedError


class _Pipeline:
    def __init__(self):
        self.calls = []

    def process_blob(self, blob, language):
        self.calls.append((blob, language))
        return None


def _write_wav(path: str, samples):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        buf = array("h", samples)
        wf.writeframes(buf.tobytes())


@pytest.fixture
def raw_wav(tmp_path):
    wav_path = tmp_path / "raw.wav"
    samples = [int(1000 * math.sin(2 * math.pi * 110 * t / 16000)) for t in range(1600)]
    _write_wav(str(wav_path), samples)
    return wav_path


def test_session_adapter_suppresses_silence(monkeypatch, raw_wav):
    pipe = _Pipeline()
    monkeypatch.setattr(sa, "build_pipeline", lambda **kwargs: pipe)
    fake_s3 = _FakeS3()

    monkeypatch.setattr(sa, "to_mono16k_pcm", lambda src, dst: shutil.copyfile(src, dst))

    silence_result = SilenceGateResult(
        is_speech=False,
        speech_ratio=0.0,
        avg_rms=10.0,
        frame_count=10,
        speech_frames=0,
        aggressiveness=2,
        reason="below_thresholds",
        frame_ms=30,
        sample_rate=16000,
        frame_bytes=960,
    )
    monkeypatch.setattr(sa, "evaluate_silence", lambda *args, **kwargs: silence_result)
    monkeypatch.setattr(sa, "config_from_env", lambda level=None: SilenceGateConfig((level or 2), 0.2, 100.0, 2))

    adapter = sa.SessionAdapter(dg_client=None, whisper_client=None, s3_client=fake_s3,
                                bucket="bucket", base_prefix="prefix")
    adapter._silence_gate_enabled = True

    result = adapter.on_segment(
        session_id="sess",
        raw_path=str(raw_wav),
        captured_ts=0.0,
        duration_s=0.5,
        language="sv",
        vad_aggressiveness=2,
    )

    assert result is None
    assert pipe.calls == []
    assert fake_s3.put_calls, "expected silence drop to be written"
    drop_payload = json.loads(fake_s3.put_calls[0]["Body"].decode("utf-8"))
    assert drop_payload["reason"] == "below_thresholds"
    assert "min_speech_ratio" in drop_payload
    assert "rms_floor" in drop_payload


def test_session_adapter_allows_speech(monkeypatch, raw_wav):
    pipe = _Pipeline()
    monkeypatch.setattr(sa, "build_pipeline", lambda **kwargs: pipe)
    fake_s3 = _FakeS3()

    monkeypatch.setattr(sa, "to_mono16k_pcm", lambda src, dst: shutil.copyfile(src, dst))

    speech_result = SilenceGateResult(
        is_speech=True,
        speech_ratio=0.6,
        avg_rms=500.0,
        frame_count=10,
        speech_frames=6,
        aggressiveness=1,
        reason="speech_ratio",
        frame_ms=30,
        sample_rate=16000,
        frame_bytes=960,
        voiced_start_frame=1,
        voiced_end_frame=4,
    )
    monkeypatch.setattr(sa, "evaluate_silence", lambda *args, **kwargs: speech_result)
    monkeypatch.setattr(sa, "config_from_env", lambda level=None: SilenceGateConfig((level or 1), 0.2, 100.0, 2))

    adapter = sa.SessionAdapter(dg_client=None, whisper_client=None, s3_client=fake_s3,
                                bucket="bucket", base_prefix="prefix")
    adapter._silence_gate_enabled = True
    calls = []
    monkeypatch.setattr(adapter, "_trim_wav_to_voiced", lambda **kwargs: calls.append(kwargs))

    result = adapter.on_segment(
        session_id="sess",
        raw_path=str(raw_wav),
        captured_ts=0.0,
        duration_s=0.5,
        language="sv",
        vad_aggressiveness=1,
    )

    assert pipe.calls, "expected pipeline to run when speech passes gate"
    assert not fake_s3.put_calls
    assert calls, "expected trimming to be invoked for voiced segment"
