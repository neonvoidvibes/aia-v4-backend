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

    def run(self, segments: List[Dict[str, Any]], mirror_markdown: str) -> str:
        # Produce Layer 2 — Lens section in Markdown grounded in Mirror
        payload = {
            "mirror_markdown": mirror_markdown,
            "segments_meta": [{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min")} for s in segments],
            "template": """
# Layer 2 — Collective Intelligence
### Lens
- pattern: <text> | systemic_significance: <text> | evidence_ids: mirror:explicit_themes:0
- tension: <text> | impact_assessment: <text> | evidence_ids: mirror:direct_quotes:0
- emotional_undercurrents: <text> | power_dynamics: <text>
- paradox: <text> | implications: <text> | evidence_ids: mirror:stated_agreements:0
"""
        }
        messages = [
            {"role": "system", "content": "Derive Lens from Mirror. Keep Markdown only. Cite evidence_ids using mirror:* indices present in the provided mirror_markdown."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=1000, temperature=0.1)
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"tx.agent.done name=LensAgent ms={dt:.1f} out_chars={len(resp or '')}")
            return resp or ""
        except Exception as e:
            logger.error(f"LensAgent LLM error: {e}")
            return "# Layer 2 — Collective Intelligence\n### Lens\n"
