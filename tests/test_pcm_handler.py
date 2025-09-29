import os
import sys
import struct

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import pcm_stream as pcm


@pytest.fixture
def session_data(tmp_path):
    return {
        "supports_pcm_stream": True,
        "pcm_frame_buffer": bytearray(),
        "pcm_accumulated_duration_ms": 0.0,
        "pcm_samples_buffered": 0,
        "pcm_segment_target_ms": 40,
        "pcm_frame_stats": {"frames_received": 0, "frames_dropped": 0, "out_of_order": 0},
    }


def _build_pcm_frame(seq: int, payload: bytes, frame_duration_ms: int = 20, frame_samples: int = 320):
    header = struct.pack(
        '<IIdHHIHHI',
        pcm.PCM_FRAME_MAGIC,
        seq,
        float(seq),
        frame_samples,
        frame_duration_ms,
        16000,
        1,
        1,
        len(payload),
    )
    return header + payload


def test_handle_pcm_frame_dispatches_when_threshold_met(monkeypatch, session_data):
    captured = {}

    def fake_dispatch(session_id, sess_data, pcm_bytes, duration_s, sample_rate, channels):
        captured["pcm_bytes"] = pcm_bytes
        captured["duration_s"] = duration_s
        captured["sample_rate"] = sample_rate
        captured["channels"] = channels

    def dispatch_fn(*args):
        fake_dispatch(*args)

    payload = (b"\x01\x00" * 320)
    frame = _build_pcm_frame(1, payload)

    assert pcm.handle_pcm_frame("sess", session_data, frame, dispatch_fn) is True
    assert not captured  # buffer not flushed yet

    frame2 = _build_pcm_frame(2, payload)
    assert pcm.handle_pcm_frame("sess", session_data, frame2, dispatch_fn) is True

    assert captured, "expected PCM segment to be dispatched"
    assert len(captured["pcm_bytes"]) == len(payload) * 2
    assert pytest.approx(captured["duration_s"], rel=1e-3) == 0.04
    assert captured["sample_rate"] == 16000
    assert captured["channels"] == 1
    assert session_data["pcm_frame_buffer"] == bytearray()
