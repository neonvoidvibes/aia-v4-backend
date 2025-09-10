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

    def run(self, segments: List[Dict[str, Any]], context_md: str | None = None) -> str:
        # Map: per-segment lens bullets
        packs: List[str] = []
        for s in segments:
            payload = {
                "segment": {"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min"), "text": s.get("text")},
                "story_context_excerpt": (context_md or "")[:2500],
                "template": """
### Lens Pack {seg_id}
- pattern: <text> | segment_ids: {seg_id}
- tension: <text> | segment_ids: {seg_id}
- emotional_undercurrents: <text> | power_dynamics: <text>
- paradox: <text> | implications: <text> | segment_ids: {seg_id}
"""
            }
            try:
                pack = chat(std_model(), [
                    {"role": "system", "content": "Extract Lens bullets for this segment, guided by the story context. Do not contradict Mirror-level facts. Markdown only. Keys in English; content in original language."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ], max_tokens=900, temperature=0.1)
                packs.append(pack or '')
            except Exception as e:
                logger.error(f"LensAgent map error seg={s.get('id')}: {e}")

        # Reduce: merge packs into a single Lens section
        reduce_payload = {"packs": packs, "story_context_excerpt": (context_md or "")[:2500], "template": """
# Layer 2 — Collective Intelligence
### Lens
- pattern: <text> | segment_ids: seg:1,seg:2
- tension: <text> | segment_ids: seg:3
- emotional_undercurrents: <text> | power_dynamics: <text>
- paradox: <text> | implications: <text> | segment_ids: seg:4
"""}
        try:
            resp = chat(std_model(), [
                {"role": "system", "content": "Merge Lens packs across all segments into a concise section, using the story context to keep coherence and tone. Markdown only."},
                {"role": "user", "content": json.dumps(reduce_payload, ensure_ascii=False)}
            ], max_tokens=1200, temperature=0.1)
            return resp or "# Layer 2 — Collective Intelligence\n### Lens\n"
        except Exception as e:
            logger.error(f"LensAgent reduce error: {e}")
            return "# Layer 2 — Collective Intelligence\n### Lens\n"
