"""Helpers for shaping transcript payloads."""
from __future__ import annotations

from typing import Dict, Optional


LOW_CONF_MARKER = "[low-confidence]"


def format_transcript_line(timestamp_str: str, text: str, metadata: Optional[Dict[str, object]] = None) -> str:
    """Format a transcript line for persistence with optional markers."""
    markers = []
    if metadata and metadata.get("low_confidence"):
        markers.append(LOW_CONF_MARKER)

    formatted_text = text
    if markers:
        formatted_text = f"{formatted_text} {' '.join(markers)}".strip()

    return f"{timestamp_str} {formatted_text}".strip()
