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

    def run(self, layer1_md: str, layer2_md: str) -> str:
        payload = {"layer1_markdown": layer1_md, "layer2_markdown": layer2_md, "template": """
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
        messages = [
            {"role": "system", "content": "Generate Layer 3 as Markdown. Use prior layer markdown; keys in English, content in original language."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=1200, temperature=0.1)
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"tx.agent.done name=WisdomAgent ms={dt:.1f} out_chars={len(resp or '')}")
            return resp or ""
        except Exception as e:
            logger.error(f"WisdomAgent LLM error: {e}")
        return "# Layer 3 — Wisdom Integration\n"
