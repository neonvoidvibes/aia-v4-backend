import logging
from typing import Optional

logger = logging.getLogger(__name__)

_vad_bridge: Optional[object] = None


def initialize_vad_bridge(openai_api_key: str):
    """VAD support disabled under the append-only transcript pipeline."""
    global _vad_bridge
    logger.info("VAD bridge initialization skipped; feature disabled.")
    _vad_bridge = None
    return _vad_bridge


def get_vad_bridge():
    return _vad_bridge


def cleanup_vad_bridge():
    logger.info("VAD bridge cleanup called (no-op).")


def is_vad_enabled() -> bool:
    return False


def log_vad_configuration():
    logger.info("VAD configuration: disabled")
