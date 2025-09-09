import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import MIRROR_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class MirrorAgent(Agent):
    name = "mirror"

    def run(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {"segments": [{"id": s.get("id"), "text": s.get("text", "")} for s in segments], "need": ["layer1", "mirror_level"]}
        messages = [
            {"role": "system", "content": MIRROR_SYS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            resp = chat(std_model(), messages, max_tokens=1600, temperature=0.1)
            data = safe_json_parse(resp)
            if isinstance(data, dict) and "layer1" in data and "mirror_level" in data:
                return data
        except Exception as e:
            logger.error(f"MirrorAgent LLM error: {e}")

        # Fallback minimal structure
        return {
            "layer1": {
                "meeting_metadata": {
                    "timestamp": "unknown",
                    "duration_minutes": max(1, len(" ".join([s.get("text", "") for s in segments])) // 1200),
                    "participants_count": 1,
                    "meeting_type": "unspecified",
                    "primary_purpose": "unspecified",
                },
                "actionable_outputs": {"decisions": [], "tasks": [], "next_meetings": []},
            },
            "mirror_level": {
                "explicit_themes": [],
                "stated_agreements": [],
                "direct_quotes": [],
                "participation_patterns": {"energy_shifts": [], "engagement_distribution": "unknown"},
            },
        }

