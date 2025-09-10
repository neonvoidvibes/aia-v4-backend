import json
import logging
from typing import Dict, Any
from .base import Agent
from ._llm import chat
from .prompts import STRATEGIC_IMPLICATIONS_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class StrategicImplicationsAgent(Agent):
    name = "strategic_implications"

    def run(self, business_reality_md: str, organizational_dynamics_md: str, context_md: str | None = None) -> str:
        """Assess strategic implications of business reality and organizational dynamics.
        Focus on business impact, alignment, risks, and opportunities.
        """
        
        payload = {
            "business_reality_content": business_reality_md[:6000],
            "organizational_dynamics_content": organizational_dynamics_md[:4000],
            "business_context": (context_md or "")[:2000]
        }
        
        try:
            implications = chat(
                std_model(),
                [
                    {"role": "system", "content": STRATEGIC_IMPLICATIONS_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2200,
                temperature=0.1,
            )
            return implications or "# Layer 3 — Strategic Implications\n(No strategic insights identified)\n"
        except Exception as e:
            logger.error(f"StrategicImplicationsAgent error: {e}")
            return "# Layer 3 — Strategic Implications\n(Error analyzing implications)\n"

    def refine(self, business_reality_md: str, organizational_dynamics_md: str, previous_output: str, feedback: str, context_md: str | None = None) -> str:
        """Refine previous strategic implications analysis based on reality check feedback."""
        
        payload = {
            "business_reality_content": business_reality_md[:4000],
            "organizational_dynamics_content": organizational_dynamics_md[:3000],
            "previous_strategic_implications": previous_output[:3500],
            "reality_check_feedback": feedback[:2500],
            "business_context": (context_md or "")[:1500]
        }
        
        try:
            from .prompts import STRATEGIC_IMPLICATIONS_REFINEMENT_SYS
            
            refined_implications = chat(
                std_model(),
                [
                    {"role": "system", "content": STRATEGIC_IMPLICATIONS_REFINEMENT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2500,
                temperature=0.1,
            )
            return refined_implications or previous_output
        except Exception as e:
            logger.error(f"StrategicImplicationsAgent refinement error: {e}")
            return previous_output
