"""Utilities for read-time windowed transcripts (no rolling files)."""
import logging
import boto3
import os
from datetime import datetime
from .rolling_transcript import RollingTranscriptWindow

def pick_latest_canonical_transcript_key(s3, bucket, base_path):
    """Find the latest canonical transcript_*.txt file."""
    try:
        if base_path.endswith('/'):
            prefix = f"{base_path}transcripts/"
        else:
            prefix = f"{base_path}/transcripts/"

        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' not in response:
            return None

        # Find canonical transcript files only
        transcript_files = [
            obj['Key'] for obj in response['Contents']
            if obj['Key'].endswith('.txt')
            and obj['Key'].rsplit('/', 1)[-1].startswith('transcript_')
            and '/transcripts/' in obj['Key']
            and '/saved/' not in obj['Key']
        ]

        if not transcript_files:
            return None

        # Return latest by LastModified
        latest = max(transcript_files, key=lambda x: s3.head_object(Bucket=bucket, Key=x)['LastModified'])
        return latest

    except Exception as e:
        logging.error(f"Error finding canonical transcript files: {e}")
        return None

def get_window_context(s3, bucket, base_path, window_seconds: int) -> str:
    """
    1) list canonical transcripts: f"{base_path}/transcripts/"
    2) pick latest 'transcript_*.txt' by LastModified
    3) read full text
    4) return windowed text via RollingTranscriptWindow.compute_window(...)
    """
    latest_key = pick_latest_canonical_transcript_key(s3, bucket, base_path)
    if not latest_key:
        return ""

    try:
        blob = s3.get_object(Bucket=bucket, Key=latest_key)['Body'].read().decode('utf-8')
        return RollingTranscriptWindow().compute_window(blob, window_seconds)
    except Exception as e:
        logging.error(f"Error reading transcript {latest_key}: {e}")
        return ""
