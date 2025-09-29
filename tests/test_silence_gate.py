import math
import os
import sys
import wave
from array import array

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.silence_gate import SilenceGateConfig, evaluate_silence, config_from_env


def _write_wav(path: str, samples, *, sample_rate: int = 16000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        buf = array("h", samples)
        wf.writeframes(buf.tobytes())


def test_silence_gate_flags_silence(tmp_path):
    wav_path = tmp_path / "silence.wav"
    _write_wav(str(wav_path), [0] * 16000)
    cfg = SilenceGateConfig(aggressiveness=1, min_speech_ratio=0.2, rms_floor=50.0, confirm_silence_windows=2)
    result = evaluate_silence(str(wav_path), config=cfg)
    assert result.is_speech is False
    assert result.reason in {"below_thresholds", "vad_unavailable"}


def test_silence_gate_allows_voiced_segments(tmp_path):
    wav_path = tmp_path / "tone.wav"
    samples = [int(2000 * math.sin(2 * math.pi * 220 * t / 16000)) for t in range(16000)]
    _write_wav(str(wav_path), samples)
    cfg = SilenceGateConfig(aggressiveness=1, min_speech_ratio=0.2, rms_floor=100.0, confirm_silence_windows=2)
    result = evaluate_silence(str(wav_path), config=cfg)
    assert result.is_speech is True
    assert result.avg_rms >= cfg.rms_floor


def test_config_from_env_respects_overrides(monkeypatch):
    monkeypatch.setenv("SILENCE_GATE_MIN_RATIO", "0.3")
    monkeypatch.setenv("SILENCE_GATE_MIN_RATIO_STEP", "0.1")
    monkeypatch.setenv("SILENCE_GATE_RMS_FLOOR", "200")
    monkeypatch.setenv("SILENCE_GATE_RMS_FLOOR_STEP", "50")
    monkeypatch.setenv("SILENCE_GATE_CONFIRM_WINDOWS", "4")
    cfg = config_from_env(3)
    assert abs(cfg.min_speech_ratio - 0.5) < 1e-6
    assert abs(cfg.rms_floor - 300.0) < 1e-6
    assert cfg.confirm_silence_windows == 4
