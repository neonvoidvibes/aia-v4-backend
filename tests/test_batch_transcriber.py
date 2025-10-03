import os
import shutil
import tempfile
import unittest
import wave
from unittest.mock import patch

from core.types import ASRResult, ASRSegment
from providers.base import ASRProvider
from utils.batch_transcriber import BatchTranscriber


def _write_silence_wav(path: str, duration_s: float, sample_rate: int = 16000) -> None:
    total_frames = int(duration_s * sample_rate)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * total_frames)


class _StubProvider(ASRProvider):
    name = "stub"

    def __init__(self) -> None:
        self.calls = []

    def transcribe_file(self, wav_path: str, language):
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        idx = len(self.calls)
        text = f"chunk-{idx}"
        self.calls.append((wav_path, duration))
        segment = ASRSegment(text=text, start_s=0.0, end_s=duration)
        return ASRResult(ok=True, segments=[segment], raw_text=text, provider=self.name, meta={})


class BatchTranscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="batch_transcriber_test_")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _make_transcriber(self, chunk_seconds: int) -> BatchTranscriber:
        stub_provider = _StubProvider()
        with patch.object(BatchTranscriber, '_resolve_providers', return_value=(stub_provider, None)):
            transcriber = BatchTranscriber(
                dg_client=None,
                whisper_client=None,
                provider_name='deepgram',
                chunk_seconds=chunk_seconds,
                max_parallel=1,
            )
        # Ensure provider stub persists on instance for assertions
        transcriber._resolve_providers = lambda: (stub_provider, None)  # type: ignore[attr-defined]
        return transcriber

    def test_segmenter_splits_by_duration(self):
        src_path = os.path.join(self.tempdir.name, 'long.wav')
        _write_silence_wav(src_path, duration_s=200.0)
        transcriber = self._make_transcriber(chunk_seconds=90)

        with patch('utils.batch_transcriber.to_mono16k_pcm', side_effect=lambda src, dst: shutil.copyfile(src, dst)):
            result = transcriber.transcribe(
                source_path=src_path,
                language='en',
                session_id='job-test',
            )

        self.assertEqual(result['chunks'], 3)
        self.assertEqual(len(result['segments']), 3)
        # Verify offsets line up with 90-second segments except final remainder
        self.assertAlmostEqual(result['segments'][0]['start'], 0.0, places=2)
        self.assertAlmostEqual(result['segments'][1]['start'], 90.0, places=0)
        self.assertTrue(result['segments'][2]['start'] >= 180.0)
        self.assertEqual(result['text'], 'chunk-0\nchunk-1\nchunk-2')

    def test_small_file_stays_single_chunk(self):
        src_path = os.path.join(self.tempdir.name, 'short.wav')
        _write_silence_wav(src_path, duration_s=45.0)
        transcriber = self._make_transcriber(chunk_seconds=90)

        with patch('utils.batch_transcriber.to_mono16k_pcm', side_effect=lambda src, dst: shutil.copyfile(src, dst)):
            result = transcriber.transcribe(
                source_path=src_path,
                language='en',
                session_id='job-test',
            )

        self.assertEqual(result['chunks'], 1)
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(result['segments'][0]['text'], 'chunk-0')
        self.assertAlmostEqual(result['segments'][0]['end'], 45.0, places=0)
        self.assertEqual(result['text'], 'chunk-0')


if __name__ == '__main__':
    unittest.main()
