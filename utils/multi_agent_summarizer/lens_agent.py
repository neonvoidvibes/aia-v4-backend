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
        # Build evidence registry for mirror anchors
        registry = []
        ml = (mirror or {}).get("mirror_level", {})
        for k in ["explicit_themes", "stated_agreements", "direct_quotes"]:
            for idx, _ in enumerate(ml.get(k, []) or []):
                registry.append(f"mirror:{k}:{idx}")

        # Only pass minimal segment meta to reduce tokens
        payload = {"segments": [{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min")} for s in segments], "mirror": mirror, "evidence_registry": registry}
        messages = [
            {"role": "system", "content": LENS_SYS + " Return JSON only. Cite evidence_ids strictly from provided registry."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=1400, temperature=0.1, response_format={"type": "json_object"})
            data = safe_json_parse(resp)
            if isinstance(data, dict) and "lens_level" in data:
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"tx.agent.done name=LensAgent ms={dt:.1f} out_chars={len(resp or '')}")
                return data
        except Exception as e:
            logger.error(f"LensAgent LLM error: {e}")

        return {"lens_level": {"hidden_patterns": [], "unspoken_tensions": [], "group_dynamics": {"emotional_undercurrents": "", "power_dynamics": ""}, "paradoxes_contradictions": []}}
