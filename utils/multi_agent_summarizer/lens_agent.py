import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import LENS_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class LensAgent(Agent):
    name = "lens"

    def run(self, segments: List[Dict[str, Any]], mirror: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"segments": [{"id": s.get("id"), "text": s.get("text", "")} for s in segments], "mirror": mirror}
        messages = [
            {"role": "system", "content": LENS_SYS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            resp = chat(std_model(), messages, max_tokens=1600, temperature=0.1)
            data = safe_json_parse(resp)
            if isinstance(data, dict) and "lens_level" in data:
                return data
        except Exception as e:
            logger.error(f"LensAgent LLM error: {e}")

        return {"lens_level": {"hidden_patterns": [], "unspoken_tensions": [], "group_dynamics": {"emotional_undercurrents": "", "power_dynamics": ""}, "paradoxes_contradictions": []}}

