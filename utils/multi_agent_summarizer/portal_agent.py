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

    def run(self, segments: Dict[str, Any], context_md: str | None = None) -> str:
        # Map: per-segment portal bullets (emergent possibilities/interventions/shifts)
        packs = []
        for s in segments:
            payload = {
                "segment": {"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min"), "text": s.get("text")},
                "story_context_excerpt": (context_md or "")[:2000],
                "template": """
### Portal Pack {seg_id}
- possibility: <text> | transformation_potential: <text> | segment_ids: {seg_id}
- leverage_point: <text> | predicted_outcomes: a, b | probability_score: 0.7 | segment_ids: {seg_id}
- shift: <text> | readiness_assessment: <text> | segment_ids: {seg_id}
"""
            }
            try:
                pack = chat(std_model(), [
                    {"role": "system", "content": "Extract Portal bullets for this segment. Use the story context to keep proposals coherent and grounded. Markdown only."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ], max_tokens=700, temperature=0.1)
                packs.append(pack or '')
            except Exception as e:
                logger.error(f"PortalAgent map error seg={s.get('id')}: {e}")

        reduce_payload = {"packs": packs, "story_context_excerpt": (context_md or "")[:2000], "template": """
# Layer 2 — Collective Intelligence
### Portal
- possibility: <text> | transformation_potential: <text> | segment_ids: seg:1,seg:2
- leverage_point: <text> | predicted_outcomes: a, b | probability_score: 0.7
- shift: <text> | readiness_assessment: <text>
"""}
        try:
            resp = chat(std_model(), [
                {"role": "system", "content": "Merge Portal packs across all segments into a concise section, aligned with the story context. Markdown only."},
                {"role": "user", "content": json.dumps(reduce_payload, ensure_ascii=False)}
            ], max_tokens=1000, temperature=0.1)
            return resp or "# Layer 2 — Collective Intelligence\n### Portal\n"
        except Exception as e:
            logger.error(f"PortalAgent reduce error: {e}")
            return "# Layer 2 — Collective Intelligence\n### Portal\n"
