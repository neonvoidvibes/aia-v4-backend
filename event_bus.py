from typing import Callable, Optional, Dict, Any

_emitter: Optional[Callable[[str, str, Dict[str, Any]], None]] = None


def set_emitter(fn: Callable[[str, str, Dict[str, Any]], None]) -> None:
    global _emitter
    _emitter = fn


def emit(session_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    if _emitter:
        try:
            _emitter(session_id, event_type, payload or {})
        except Exception:
            pass
