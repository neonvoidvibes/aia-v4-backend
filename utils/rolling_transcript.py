"""Rolling transcript window computation (no file writes)."""
import os
import boto3
import logging
import re
from datetime import datetime, timezone, timedelta
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
        now = datetime.now(timezone.utc)
        window_delta = timedelta(seconds=window_seconds)
        filtered_lines = []

        for line in lines:
            # Skip headers and metadata lines
            if line.startswith('#') or not line.strip():
                continue

            try:
                # Extract timestamp from line format "[HH:MM:SS - ...]" or "[HH:MM:SS.mmm UTC]"
                timestamp_match = re.search(r'\[([0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{3})?)', line)
                if not timestamp_match:
                    continue

                timestamp_str = timestamp_match.group(1)

                # Parse timestamp - assume it's from today for simplicity
                today = now.date()
                if '.' in timestamp_str:
                    # Format: HH:MM:SS.mmm
                    time_part = datetime.strptime(timestamp_str, "%H:%M:%S.%f").time()
                else:
                    # Format: HH:MM:SS
                    time_part = datetime.strptime(timestamp_str, "%H:%M:%S").time()

                timestamp = datetime.combine(today, time_part).replace(tzinfo=timezone.utc)

                # Keep lines within the window
                if now - timestamp <= window_delta:
                    filtered_lines.append(line)

            except Exception as e:
                logger.warning(f"Error parsing timestamp in line: {line[:50]}... Error: {e}")
                continue

        return "\n".join(filtered_lines)
