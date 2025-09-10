import json
import logging
from typing import Dict, Any
from .base import Agent
from ._llm import chat
from .prompts import NEXT_ACTIONS_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class NextActionsAgent(Agent):
    name = "next_actions"

    def run(self, business_reality_md: str, organizational_dynamics_md: str, strategic_implications_md: str, context_md: str | None = None) -> str:
        """Generate concrete, actionable next steps based on all previous analysis.
        Focus on specific actions that can be taken in the next 1-4 weeks.
        """
        
        payload = {
            "business_reality_content": business_reality_md[:4000],
            "organizational_dynamics_content": organizational_dynamics_md[:3000], 
            "strategic_implications_content": strategic_implications_md[:3000],
            "business_context": (context_md or "")[:1500]
        }
        
        try:
            actions = chat(
                std_model(),
                [
                    {"role": "system", "content": NEXT_ACTIONS_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            return actions or "# Layer 4 — Next Actions\n(No actions identified)\n"
        except Exception as e:
            logger.error(f"NextActionsAgent error: {e}")
            return "# Layer 4 — Next Actions\n(Error generating actions)\n"