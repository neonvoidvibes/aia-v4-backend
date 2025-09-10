import json
import logging
from typing import Dict, Any
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import LEARNING_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class LearningAgent(Agent):
    name = "learning"

    def run(self, layer1: Dict[str, Any], layer2: Dict[str, Any], layer3: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"layer1": layer1, "layer2": layer2, "layer3": layer3}
        messages = [
            {"role": "system", "content": LEARNING_SYS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=1400, temperature=0.1, response_format={"type": "json_object"})
            data = safe_json_parse(resp)
            if isinstance(data, dict) and "layer4" in data:
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"tx.agent.done name=LearningAgent ms={dt:.1f} out_chars={len(resp or '')}")
                return data["layer4"] if "triple_loop_learning" in data["layer4"] else data
        except Exception as e:
            logger.error(f"LearningAgent LLM error: {e}")

        return {
            "triple_loop_learning": {"single_loop": [], "double_loop": [], "triple_loop": []},
            "warm_data_patterns": {"relational_insights": [], "transcontextual_connections": [], "living_systems_recognition": []},
            "knowledge_evolution": {"insights_captured": [], "wisdom_moments": [], "capacity_building": []},
        }
