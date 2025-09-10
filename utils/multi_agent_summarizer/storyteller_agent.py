import json
import logging
from typing import List, Dict, Any

from .base import Agent
from ._llm import chat
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class StorytellerAgent(Agent):
    name = "storyteller"

    def run(self, segments: List[Dict[str, Any]]) -> str:
        """Create a narrative prologue that sets context, stakes, and tone.
        Returns markdown only. Keys in English; content preserves original language.
        """
        # Map: generate short vignettes per segment to keep reduce grounded
        vignettes: List[str] = []
        for s in segments:
            payload = {
                "segment": {
                    "id": s.get("id"),
                    "start_min": s.get("start_min"),
                    "end_min": s.get("end_min"),
                    "text": s.get("text"),
                },
                "template": """
### Vignette {seg_id}
- scene: <where we meet the group>
- central_figures: <roles or names>
- stakes: <what matters now>
- turning_points: <moments that shift direction>
- tone_cues: <emotions, tempo>
- artifacts: <documents, tools, constraints>
- open_questions: <the living questions>
""",
            }
            try:
                vignette = chat(
                    std_model(),
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a Storyteller. Write a compact vignette for this single segment. "
                                "Return Markdown only. Use English keys; preserve original language for content."
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    max_tokens=700,
                    temperature=0.2,
                )
                vignettes.append(vignette or "")
            except Exception as e:
                logger.error(f"StorytellerAgent vignette error seg={s.get('id')}: {e}")

        # Reduce: weave vignettes into a single prologue
        reduce_payload = {
            "vignettes": vignettes,
            "template": """
# Prologue — Storyteller
### Scene & Stakes
- context: <what world are we in?>
- stakes: <why it matters now>

### Characters & Roles
- key_actors: <roles, names, relationships>
- responsibilities: <who holds what>

### Narrative Arc
- beginning: <setup>
- middle: <conflicts and developments>
- end: <where we land for now>

### Tensions & Hopes
- tensions: <frictions, tradeoffs>
- hopes: <aspirations, desired outcomes>

### Vocabulary & Glossary
- terms: <domain words, acronyms>

### Voice & Frame
- narrator_voice: <tone, distance>
- frame: <lens through which to interpret>

### What Matters Now
- focus: <what to attend to next>
""",
        }
        try:
            prologue = chat(
                std_model(),
                [
                    {
                        "role": "system",
                        "content": (
                            "Weave the vignettes into a cohesive story prologue that sets context for analysis. "
                            "Markdown only. Keep it grounded and concise; avoid repetition."
                        ),
                    },
                    {"role": "user", "content": json.dumps(reduce_payload, ensure_ascii=False)},
                ],
                max_tokens=1400,
                temperature=0.2,
            )
            return prologue or "# Prologue — Storyteller\n"
        except Exception as e:
            logger.error(f"StorytellerAgent reduce error: {e}")
            return "# Prologue — Storyteller\n"

