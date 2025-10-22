"""
Helpers for working with workspace-level settings pulled from Supabase.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_AGENT_WORKSPACE_CACHE: Dict[str, Tuple[Optional[str], datetime]] = {}
_WORKSPACE_UI_CACHE: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
_CACHE_TTL = timedelta(minutes=10)
_CACHE_TTL_FAILURE = timedelta(minutes=1)


def _cache_valid(expiry: datetime) -> bool:
    return expiry > datetime.now(timezone.utc)


def get_agent_workspace_id(agent_name: Optional[str]) -> Optional[str]:
    """
    Resolve the workspace ID for a given agent name (cached).
    """
    if not agent_name:
        return None

    cached = _AGENT_WORKSPACE_CACHE.get(agent_name)
    if cached and _cache_valid(cached[1]):
        return cached[0]

    client = get_supabase_client()
    if not client:
        logger.warning("Supabase unavailable when resolving workspace for agent '%s'", agent_name)
        return None

    try:
        response = (
            client.table("agents")
            .select("workspace_id")
            .eq("name", agent_name)
            .limit(1)
            .execute()
        )
        if getattr(response, "error", None):
            logger.error(
                "Error fetching workspace_id for agent '%s': %s",
                agent_name,
                response.error,
            )
            return None

        workspace_id = None
        if response.data:
            workspace_id = response.data[0].get("workspace_id")

        if workspace_id:
            _AGENT_WORKSPACE_CACHE[agent_name] = (
                workspace_id,
                datetime.now(timezone.utc) + _CACHE_TTL,
            )

        return workspace_id
    except Exception as exc:
        logger.warning(
            "Unexpected error fetching workspace_id for agent '%s': %s",
            agent_name,
            exc,
        )
        _AGENT_WORKSPACE_CACHE[agent_name] = (
            None,
            datetime.now(timezone.utc) + _CACHE_TTL_FAILURE,
        )
        return None


def get_workspace_ui_config(workspace_id: Optional[str]) -> Dict[str, Any]:
    """
    Fetch (and cache) the workspace ui_config JSON for the given workspace.
    Returns an empty dict when not found or on errors.
    """
    if not workspace_id:
        return {}

    cached = _WORKSPACE_UI_CACHE.get(workspace_id)
    if cached and _cache_valid(cached[1]):
        return cached[0]

    client = get_supabase_client()
    if not client:
        logger.warning("Supabase unavailable when fetching ui_config for workspace '%s'", workspace_id)
        return {}

    try:
        response = (
            client.table("workspaces")
            .select("ui_config")
            .eq("id", workspace_id)
            .limit(1)
            .execute()
        )
        if getattr(response, "error", None):
            logger.error(
                "Error fetching ui_config for workspace '%s': %s",
                workspace_id,
                response.error,
            )
            return {}

        ui_config = {}
        if response.data:
            raw_config = response.data[0].get("ui_config")
            if isinstance(raw_config, dict):
                ui_config = raw_config
            elif isinstance(raw_config, str) and raw_config.strip():
                try:
                    ui_config = json.loads(raw_config)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON in ui_config for workspace '%s': %s",
                        workspace_id,
                        raw_config[:200],
                    )

        _WORKSPACE_UI_CACHE[workspace_id] = (
            ui_config,
            datetime.now(timezone.utc) + _CACHE_TTL,
        )
        return ui_config
    except Exception as exc:
        logger.warning(
            "Unexpected error fetching ui_config for workspace '%s': %s",
            workspace_id,
            exc,
        )
        _WORKSPACE_UI_CACHE[workspace_id] = (
            {},
            datetime.now(timezone.utc) + _CACHE_TTL_FAILURE,
        )
        return {}


def memorized_transcript_scoping_enabled(
    agent_name: Optional[str],
    workspace_id: Optional[str] = None,
) -> bool:
    """
    Determine whether the "scoped memorized transcript" behaviour is enabled.

    Enabled by default; workspaces can opt out via ui_config flag
    `disable_memorized_transcript_scoping`.
    """
    if not agent_name and not workspace_id:
        return True

    resolved_workspace_id = workspace_id or get_agent_workspace_id(agent_name)
    if not resolved_workspace_id:
        return True

    ui_config = get_workspace_ui_config(resolved_workspace_id)
    disable_flag = ui_config.get("disable_memorized_transcript_scoping")
    if isinstance(disable_flag, bool):
        return not disable_flag
    if isinstance(disable_flag, str):
        normalized = disable_flag.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return False

    return True


def clear_workspace_cache() -> None:
    """
    Helper to clear internal caches (useful for tests).
    """
    _AGENT_WORKSPACE_CACHE.clear()
    _WORKSPACE_UI_CACHE.clear()
