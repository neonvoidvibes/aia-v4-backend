"""Helpers for working with WebM container headers."""

from __future__ import annotations

from typing import Optional


_CLUSTER_ID = b"\x1fC\xb6u"  # EBML ID for Cluster elements (audio payload lives here)


def extract_webm_header(blob: bytes) -> Optional[bytes]:
    """Return only the WebM container header bytes from the first blob.

    The header is everything before the first Cluster element. If the Cluster
    marker cannot be located, we return ``None`` so callers can fall back to the
    legacy behaviour.
    """

    if not blob:
        return None

    cluster_idx = blob.find(_CLUSTER_ID)
    if cluster_idx <= 0:
        # No cluster present (or header-less blob). Treat as parse failure.
        return None

    header = blob[:cluster_idx]
    # Guard against suspiciously short headers (likely a false positive).
    if len(header) < 4:
        return None

    return header

