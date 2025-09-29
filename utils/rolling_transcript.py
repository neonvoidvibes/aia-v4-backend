"""Rolling transcript window computation (no file writes)."""
import os
import boto3
import logging
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List
from dateutil import parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RollingTranscriptWindow:
    """Compute windowed transcript content at read-time (no S3 writes)."""

    def compute_window(self, full_text: str, window_seconds: int) -> str:
        """
        Pure function: given the full transcript text and a window size,
        return only lines within the last `window_seconds` based on [HH:MM:SS] stamps.
        No S3 writes.
        """
        if not full_text.strip():
            return ""

        lines = full_text.splitlines()
        now_utc = datetime.now(timezone.utc)
        window_delta = timedelta(seconds=window_seconds)
        filtered_lines = []

        session_tzinfo = timezone.utc

        for header_line in lines:
            if header_line.lower().startswith('session timezone:'):
                tz_value = header_line.split(':', 1)[1].strip()
                if tz_value:
                    tz_name = tz_value.split('(', 1)[0].strip() or 'UTC'
                    try:
                        session_tzinfo = ZoneInfo(tz_name)
                    except Exception:
                        logger.warning(f"Unknown session timezone '{tz_name}', defaulting to UTC")
                        session_tzinfo = timezone.utc
                break

        now_local = now_utc.astimezone(session_tzinfo)
        today_local = now_local.date()

        for line in lines:
            # Skip headers and metadata lines
            if line.startswith('#') or not line.strip():
                continue

            try:
                timestamp_match = re.search(r'\[([0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{3})?)(?:\s+[^\]]+)?\]', line)
                if not timestamp_match:
                    continue

                timestamp_str = timestamp_match.group(1)

                if '.' in timestamp_str:
                    time_part = datetime.strptime(timestamp_str, "%H:%M:%S.%f").time()
                else:
                    time_part = datetime.strptime(timestamp_str, "%H:%M:%S").time()

                timestamp_local = datetime.combine(today_local, time_part).replace(tzinfo=session_tzinfo)
                timestamp_utc = timestamp_local.astimezone(timezone.utc)

                if now_utc - timestamp_utc <= window_delta:
                    filtered_lines.append(line)

            except Exception as e:
                logger.warning(f"Error parsing timestamp in line: {line[:50]}... Error: {e}")
                continue

        return "\n".join(filtered_lines)
