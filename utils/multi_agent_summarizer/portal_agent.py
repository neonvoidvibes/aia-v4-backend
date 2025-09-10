import json
import logging
from typing import Dict, Any
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import PORTAL_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class PortalAgent(Agent):
    name = "portal"

    def run(self, mirror: Dict[str, Any], lens: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"mirror": mirror, "lens": lens}
        messages = [
            {"role": "system", "content": PORTAL_SYS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=1200, temperature=0.1)
            data = safe_json_parse(resp)
            if isinstance(data, dict) and "portal_level" in data:
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"tx.agent.done name=PortalAgent ms={dt:.1f} out_chars={len(resp or '')}")
                return data
        except Exception as e:
            logger.error(f"PortalAgent LLM error: {e}")

        return {"portal_level": {"emergent_possibilities": [], "intervention_opportunities": [], "paradigm_shifts": []}}
