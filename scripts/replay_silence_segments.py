#!/usr/bin/env python3
"""Inspect and optionally re-run silence-gated segments for a session."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, Dict, Any

import boto3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.silence_gate import evaluate_silence


def _iter_drop_records(client, bucket: str, prefix: str) -> Iterable[Dict[str, Any]]:
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.json'):
                continue
            response = client.get_object(Bucket=bucket, Key=key)
            payload = json.loads(response['Body'].read().decode('utf-8'))
            payload['s3_key'] = key
            yield payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect silence drops in the WAL for a session")
    parser.add_argument('--session', required=True, help='Transcript session ID (hex)')
    parser.add_argument('--bucket', help='Override transcript bucket (defaults to TRANSCRIPT_BUCKET or AWS_S3_BUCKET)')
    parser.add_argument('--prefix', help='Override transcript prefix (defaults to TRANSCRIPT_PREFIX)')
    parser.add_argument('--rerun-gate', action='store_true', help='Re-run the silence gate locally if wav path exists')
    args = parser.parse_args()

    bucket = args.bucket or os.getenv('TRANSCRIPT_BUCKET') or os.getenv('AWS_S3_BUCKET')
    if not bucket:
        raise SystemExit('No transcript bucket configured. Set TRANSCRIPT_BUCKET or AWS_S3_BUCKET.')

    base_prefix = args.prefix or os.getenv('TRANSCRIPT_PREFIX') or 'organizations/river/agents'
    drop_prefix = f"{base_prefix.rstrip('/')}/{args.session}/drops/"

    client = boto3.client('s3')

    found = False
    for record in _iter_drop_records(client, bucket, drop_prefix):
        found = True
        local_path = record.get('local_wav_path') or '-'
        print(f"seq={record['seq']:012d} ratio={record['speech_ratio']:.3f} rms={record['avg_rms']:.1f} "
              f"reason={record['reason']} aggr={record.get('aggressiveness')} wav={local_path}")
        if args.rerun_gate and local_path and local_path != '-' and os.path.exists(local_path):
            rerun = evaluate_silence(local_path, aggressiveness=record.get('aggressiveness'))
            status = 'speech' if rerun.is_speech else 'silence'
            print(f"  ↳ re-run gate: {status} (ratio={rerun.speech_ratio:.3f}, rms={rerun.avg_rms:.1f}, reason={rerun.reason})")
        elif args.rerun_gate and (not local_path or not os.path.exists(local_path)):
            print("  ↳ re-run skipped (local wav not found)")

    if not found:
        print(f"No silence-drop records found under s3://{bucket}/{drop_prefix}")


if __name__ == '__main__':
    main()
