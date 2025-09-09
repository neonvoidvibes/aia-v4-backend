import json
import logging
from typing import Dict, Any
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import WISDOM_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class WisdomAgent(Agent):
    name = "wisdom"

    def run(self, layer1: Dict[str, Any], layer2: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"layer1": layer1, "layer2": layer2}
        messages = [
            {"role": "system", "content": WISDOM_SYS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            resp = chat(std_model(), messages, max_tokens=1400, temperature=0.1)
            data = safe_json_parse(resp)
            if isinstance(data, dict) and "layer3" in data:
                return data["layer3"] if "connectedness_patterns" in data["layer3"] else data
        except Exception as e:
            logger.error(f"WisdomAgent LLM error: {e}")

        return {
            "connectedness_patterns": {
                "self_awareness": [],
                "interpersonal_dynamics": [],
                "systemic_understanding": [],
            },
            "transcendental_alignment": {
                "beauty_moments": [],
                "truth_emergence": [],
                "goodness_orientation": [],
                "coherence_quality": "",
            },
            "sovereignty_development": {
                "sentience_expansion": [],
                "intelligence_integration": [],
                "agency_manifestation": [],
            },
        }

