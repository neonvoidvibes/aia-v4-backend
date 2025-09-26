"""Utility helpers for lightweight feature flag checks."""
from __future__ import annotations

import os
from functools import lru_cache


def _flag_env_key(flag_name: str) -> str:
    safe = flag_name.replace(".", "_").replace("-", "_")
    return f"FEATURE_{safe.upper()}"


@lru_cache(maxsize=None)
def feature_enabled(flag_name: str, default: bool = False) -> bool:
    """Return True when the feature flag is enabled via environment variable.

    Flags are looked up using the pattern ``FEATURE_<FLAG_NAME>`` where dots and
    dashes are converted to underscores. Truthy values: 1, true, yes, on.
    """
    env_key = _flag_env_key(flag_name)
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def reset_flag_cache() -> None:
    """Clear the cached feature flag lookups (useful in tests)."""
    feature_enabled.cache_clear()
