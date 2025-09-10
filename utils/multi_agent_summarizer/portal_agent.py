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
        # Build lens registry
        lens_registry = []
        ll = (lens or {}).get("lens_level", {})
        for k in ["hidden_patterns", "unspoken_tensions", "paradoxes_contradictions"]:
            for idx, _ in enumerate(ll.get(k, []) or []):
                lens_registry.append(f"lens:{k}:{idx}")
        payload = {"mirror": mirror, "lens": lens, "lens_registry": lens_registry}
        messages = [
            {"role": "system", "content": PORTAL_SYS + " Return JSON only. evidence_ids must be drawn from lens_registry."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=1200, temperature=0.1, response_format={"type": "json_object"})
            data = safe_json_parse(resp)
            if isinstance(data, dict) and "portal_level" in data:
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"tx.agent.done name=PortalAgent ms={dt:.1f} out_chars={len(resp or '')}")
                return data
        except Exception as e:
            logger.error(f"PortalAgent LLM error: {e}")

        return {"portal_level": {"emergent_possibilities": [], "intervention_opportunities": [], "paradigm_shifts": []}}
