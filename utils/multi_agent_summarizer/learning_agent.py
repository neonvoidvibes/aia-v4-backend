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

    def run(self, segments: Dict[str, Any], context_md: str | None = None) -> str:
        # Map: per-segment learning bullets
        packs = []
        for s in segments:
            payload = {
                "segment": {"id": s.get("id"), "text": s.get("text")},
                "story_context_excerpt": (context_md or "")[:2000],
                "template": """
### Learning Pack {seg_id}
- error_correction: <text> | process_improvement: <text>
- assumption_questioning: <text> | mental_model_shift: <text>
- context_examination: <text> | paradigm_transformation: <text>
- relationship_between: a, b | pattern: <text> | systemic_impact: <text>
- contexts: a, b | emergent_property: <text>
- system_characteristic: <text> | health_indicator: <text>
- insight: <text> | application_potential: <text> | integration_path: <text>
- wisdom_expression: <text> | depth_indicator: <text> | collective_impact: <text>
- capacity: <text> | development_trajectory: <text>
"""}
            try:
                pack = chat(std_model(), [
                    {"role": "system", "content": "Extract Layer 4 (Learning) bullets for this segment, keeping alignment with the story context. Markdown only."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ], max_tokens=1100, temperature=0.1)
                packs.append(pack or '')
            except Exception as e:
                logger.error(f"LearningAgent map error seg={s.get('id')}: {e}")

        reduce_payload = {"packs": packs, "story_context_excerpt": (context_md or "")[:2000], "template": """
# Layer 4 — Learning & Development
### Triple Loop Learning
- error_correction: <text> | process_improvement: <text>
- assumption_questioning: <text> | mental_model_shift: <text>
- context_examination: <text> | paradigm_transformation: <text>
### Warm Data Patterns
- relationship_between: a, b | pattern: <text> | systemic_impact: <text>
- contexts: a, b | emergent_property: <text>
- system_characteristic: <text> | health_indicator: <text>
### Knowledge Evolution
- insight: <text> | application_potential: <text> | integration_path: <text>
- wisdom_expression: <text> | depth_indicator: <text> | collective_impact: <text>
- capacity: <text> | development_trajectory: <text>
"""}
        try:
            resp = chat(std_model(), [
                {"role": "system", "content": "Merge Learning packs across all segments into a concise section, with tone guided by the story context. Markdown only."},
                {"role": "user", "content": json.dumps(reduce_payload, ensure_ascii=False)}
            ], max_tokens=1400, temperature=0.1)
            return resp or "# Layer 4 — Learning & Development\n"
        except Exception as e:
            logger.error(f"LearningAgent reduce error: {e}")
            return "# Layer 4 — Learning & Development\n"
