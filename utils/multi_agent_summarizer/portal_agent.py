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

    def run(self, mirror_markdown: str, lens_markdown: str) -> str:
        # Produce Layer 2 — Portal section in Markdown grounded in Lens
        payload = {
            "mirror_markdown": mirror_markdown,
            "lens_markdown": lens_markdown,
            "template": """
# Layer 2 — Collective Intelligence
### Portal
- possibility: <text> | transformation_potential: <text> | evidence_ids: lens:hidden_patterns:0
- leverage_point: <text> | predicted_outcomes: a, b | probability_score: 0.7 | evidence_ids: lens:unspoken_tensions:0
- shift: <text> | readiness_assessment: <text> | evidence_ids: lens:paradoxes_contradictions:0
"""
        }
        messages = [
            {"role": "system", "content": "Derive Portal from Mirror+Lens. Markdown only. Cite evidence_ids using lens:* indices present in the provided lens_markdown."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=900, temperature=0.1)
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"tx.agent.done name=PortalAgent ms={dt:.1f} out_chars={len(resp or '')}")
            return resp or ""
        except Exception as e:
            logger.error(f"PortalAgent LLM error: {e}")
            return "# Layer 2 — Collective Intelligence\n### Portal\n"
