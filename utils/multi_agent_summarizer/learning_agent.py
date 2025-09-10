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

    def run(self, layer1_md: str, layer2_md: str, layer3_md: str) -> str:
        payload = {"layer1_markdown": layer1_md, "layer2_markdown": layer2_md, "layer3_markdown": layer3_md, "template": """
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
        messages = [
            {"role": "system", "content": "Generate Layer 4 as Markdown from prior layers. Keys in English; content in original language."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            import time
            t0 = time.perf_counter()
            resp = chat(std_model(), messages, max_tokens=1200, temperature=0.1)
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"tx.agent.done name=LearningAgent ms={dt:.1f} out_chars={len(resp or '')}")
            return resp or ""
        except Exception as e:
            logger.error(f"LearningAgent LLM error: {e}")
            return "# Layer 4 — Learning & Development\n"
