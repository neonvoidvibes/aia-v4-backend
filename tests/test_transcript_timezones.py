from __future__ import annotations

import io
import os
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.session_adapter import SessionAdapter
from utils.rolling_transcript import RollingTranscriptWindow


class _StubS3Client:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> None:
        self._objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        try:
            data = self._objects[(Bucket, Key)]
        except KeyError as exc:
            raise self.exceptions.NoSuchKey() from exc
        return {"Body": io.BytesIO(data)}


class _StubASRClient:
    def transcribe_file(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"results": {"channels": []}}


def _make_adapter() -> SessionAdapter:
    return SessionAdapter(
        dg_client=_StubASRClient(),
        whisper_client=_StubASRClient(),
        s3_client=_StubS3Client(),
        bucket="stub-bucket",
        base_prefix="stub-prefix",
    )


def test_format_timestamp_with_timezone() -> None:
    adapter = _make_adapter()
    session_id = "session-paris"
    adapter.register_session(session_id, "key", "Europe/Paris")

    captured_utc = datetime(2024, 7, 1, 8, 0, 0, tzinfo=timezone.utc)
    formatted = adapter._format_timestamp(session_id, captured_utc.timestamp())

    assert formatted.startswith("[10:00:00")
    assert formatted.endswith("CEST]")


def test_format_timestamp_defaults_to_utc() -> None:
    adapter = _make_adapter()
    session_id = "session-utc"
    adapter.register_session(session_id, "key")

    captured_utc = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    formatted = adapter._format_timestamp(session_id, captured_utc.timestamp())

    assert formatted == "[08:00:00 UTC]"


def test_rolling_window_respects_session_timezone() -> None:
    tz = ZoneInfo("Europe/Paris")
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)

    recent_local = now_local - timedelta(seconds=30)
    old_local = now_local - timedelta(minutes=5)

    transcript_text = "\n".join(
        [
            "# Transcript - Session abc",
            "Session Timezone: Europe/Paris (CEST)",
            "",
            f"[{old_local.strftime('%H:%M:%S')} {old_local.tzname()}] Old line",
            f"[{recent_local.strftime('%H:%M:%S')} {recent_local.tzname()}] Recent line",
        ]
    )

    window = RollingTranscriptWindow()
    result = window.compute_window(transcript_text, window_seconds=120)

    assert "Recent line" in result
    assert "Old line" not in result
