"""
Session state model with provider control and pause/resume signaling.
"""
from typing import Optional, Literal
from pydantic import BaseModel
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

    class Config:
        # Allow mutation for runtime updates
        validate_assignment = True