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
                temperature=0.3,
            )
            return actions or "# Layer 4 — Next Actions\n(No actions identified)\n"
        except Exception as e:
            logger.error(f"NextActionsAgent error: {e}")
            return "# Layer 4 — Next Actions\n(Error generating actions)\n"

    def refine(self, business_reality_md: str, organizational_dynamics_md: str, strategic_implications_md: str, previous_output: str, feedback: str, context_md: str | None = None) -> str:
        """Refine previous next actions analysis based on reality check feedback."""
        
        payload = {
            "business_reality_content": business_reality_md[:3500],
            "organizational_dynamics_content": organizational_dynamics_md[:2500], 
            "strategic_implications_content": strategic_implications_md[:2500],
            "previous_next_actions": previous_output[:3000],
            "reality_check_feedback": feedback[:2000],
            "business_context": (context_md or "")[:1200]
        }
        
        try:
            from .prompts import NEXT_ACTIONS_REFINEMENT_SYS
            
            refined_actions = chat(
                std_model(),
                [
                    {"role": "system", "content": NEXT_ACTIONS_REFINEMENT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2200,
                temperature=0.3,
            )
            return refined_actions or previous_output
        except Exception as e:
            logger.error(f"NextActionsAgent refinement error: {e}")
            return previous_output