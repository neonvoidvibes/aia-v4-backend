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

    def run(self, segments: Dict[str, Any], context_md: str | None = None) -> str:
        # Map: per-segment wisdom bullets
        packs = []
        for s in segments:
            payload = {
                "segment": {"id": s.get("id"), "text": s.get("text")},
                "story_context_excerpt": (context_md or "")[:2000],
                "template": """
### Wisdom Pack {seg_id}
- insight: <text> | individual_growth: <text>
- relationship_shift: <text> | collaboration_quality: <text>
- systems_insight: <text> | broader_context: <text>
- aesthetic_insight: <text> | elegance_factor: <text>
- truth_revealed: <text> | reality_alignment: <text>
- life_affirming_direction: <text> | stakeholder_benefit: <text>
- awareness_deepening: <text> | empathy_development: <text>
- sense_making_advancement: <text> | complexity_navigation: <text>
- responsibility_taking: <text> | purposeful_action: <text>
"""}
            try:
                pack = chat(std_model(), [
                    {"role": "system", "content": "Extract Layer 3 (Wisdom) bullets for this segment, using the story context to hold coherence. Markdown only."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ], max_tokens=1100, temperature=0.1)
                packs.append(pack or '')
            except Exception as e:
                logger.error(f"WisdomAgent map error seg={s.get('id')}: {e}")

        reduce_payload = {"packs": packs, "story_context_excerpt": (context_md or "")[:2000], "template": """
# Layer 3 — Wisdom Integration
### Connectedness Patterns
- insight: <text> | individual_growth: <text>
- relationship_shift: <text> | collaboration_quality: <text>
- systems_insight: <text> | broader_context: <text>
### Transcendental Alignment
- aesthetic_insight: <text> | elegance_factor: <text>
- truth_revealed: <text> | reality_alignment: <text>
- life_affirming_direction: <text> | stakeholder_benefit: <text>
- coherence_quality: <text>
### Sovereignty Development
- awareness_deepening: <text> | empathy_development: <text>
- sense_making_advancement: <text> | complexity_navigation: <text>
- responsibility_taking: <text> | purposeful_action: <text>
"""}
        try:
            resp = chat(std_model(), [
                {"role": "system", "content": "Merge Wisdom packs across all segments into a concise section. Keep tone aligned with the story context. Markdown only."},
                {"role": "user", "content": json.dumps(reduce_payload, ensure_ascii=False)}
            ], max_tokens=1400, temperature=0.1)
            return resp or "# Layer 3 — Wisdom Integration\n"
        except Exception as e:
            logger.error(f"WisdomAgent reduce error: {e}")
            return "# Layer 3 — Wisdom Integration\n"
