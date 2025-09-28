from __future__ import annotations
from typing import Optional
from core.state_machine import TranscriptionPipeline
from providers.deepgram_provider import DeepgramProvider
from providers.whisper_provider import WhisperProvider
from storage.log_store import WriteAheadLog
from core.types import TranscriptChunk

# Replace with your actual SDK clients and S3 client

def build_pipeline(*, dg_client, whisper_client, s3_client, bucket: str, base_prefix: str) -> TranscriptionPipeline:
    wal = WriteAheadLog(s3_client, bucket=bucket, base_prefix=base_prefix)
    p1 = DeepgramProvider(dg_client)
    p2 = WhisperProvider(whisper_client)

    def wal_append(chunk: TranscriptChunk) -> str:
        return wal.append_chunk(chunk)

    def on_commit(chunk: TranscriptChunk, key: str) -> None:
        # Optional: metrics, websockets notifications, etc. No content mutation here.
        pass

    return TranscriptionPipeline(provider_primary=p1, provider_fallback=p2,
                                 wal_append=wal_append, on_chunk_committed=on_commit)

