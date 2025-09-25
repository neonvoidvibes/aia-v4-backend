"""
Session state model with provider control and pause/resume signaling.
"""
from typing import Optional, Literal
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
import time


class SessionState(BaseModel):
    id: str
    user_id: str
    language_hint: Optional[str] = None

    # provider control
    provider_current: Literal["deepgram", "whisper"] = "deepgram"
    provider_fallback_at: Optional[float] = None      # epoch when we fell back
    provider_retry_after: Optional[float] = None      # epoch when we may probe DG again
    provider_probe_inflight: bool = False
    provider_cooldown_sec: int = 900                  # default 15 min; configurable

    # transient pause/resume signaling
    net_paused: bool = False
    last_pause_sent_at: Optional[float] = None

    # WebSocket reattachment state
    ws_connected: bool = False
    reattach_deadline: Optional[datetime] = None
    last_status: Optional[dict] = None  # Store last status for polling endpoints

    def mark_ws_disconnected(self, grace_seconds: int) -> None:
        """Mark WebSocket as disconnected and set reattachment deadline."""
        self.ws_connected = False
        self.reattach_deadline = datetime.now(timezone.utc) + timedelta(seconds=grace_seconds)

    def mark_ws_reconnected(self) -> None:
        """Mark WebSocket as reconnected and clear reattachment deadline."""
        self.ws_connected = True
        self.reattach_deadline = None

    def is_reattach_expired(self) -> bool:
        """Check if reattachment grace period has expired."""
        if not self.reattach_deadline:
            return False
        return datetime.now(timezone.utc) > self.reattach_deadline

    class Config:
        # Allow mutation for runtime updates
        validate_assignment = True