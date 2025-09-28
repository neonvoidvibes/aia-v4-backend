from __future__ import annotations
import json, os, io
from typing import List, Dict, Any
from core.types import TranscriptChunk

# This module provides an append-only write-ahead log (WAL) and a manifest.
# WAL entries are immutable small JSONL files; manifest is a tiny index.

class WriteAheadLog:
    def __init__(self, bucket_client, bucket: str, base_prefix: str) -> None:
        self._s3 = bucket_client
        self._bucket = bucket
        self._base = base_prefix.rstrip("/")  # e.g. organizations/.../transcripts

    def _chunk_key(self, session_id: str, seq: int) -> str:
        return f"{self._base}/{session_id}/chunks/{seq:012d}.json"

    def _manifest_key(self, session_id: str) -> str:
        return f"{self._base}/{session_id}/manifest.json"

    def append_chunk(self, chunk: TranscriptChunk) -> str:
        key = self._chunk_key(chunk.session_id, chunk.seq)
        body = json.dumps({
            "session_id": chunk.session_id,
            "seq": chunk.seq,
            "captured_ts": chunk.captured_ts,
            "text": chunk.text,
            "provider": chunk.provider,
            "meta": chunk.meta,
            "byte_len": chunk.byte_len,
            "created_ts": chunk.created_ts,
        }, ensure_ascii=False).encode("utf-8")
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body, ContentType="application/json")
        return key

    def publish_manifest(self, session_id: str) -> str:
        """
        Build a sparse manifest from actually present chunk objects.
        Never assume contiguous seqs; list and parse.
        """
        prefix = f"{self._base}/{session_id}/chunks/"
        paginator = self._s3.get_paginator("list_objects_v2")
        items = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                filename = key.rsplit("/", 1)[-1]
                try:
                    seq_str = filename[:-5]
                    seq = int(seq_str)
                except Exception:
                    continue
                items.append({"seq": seq, "key": key})
        items.sort(key=lambda x: x["seq"])

        manifest = {
            "session_id": session_id,
            "version": 1,
            "chunks": items,
        }
        key = self._manifest_key(session_id)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        return key

