import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat
from .prompts import ORGANIZATIONAL_DYNAMICS_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class OrganizationalDynamicsAgent(Agent):
    name = "organizational_dynamics"

    def run(self, business_reality_md: str, context_md: str | None = None) -> str:
        """Identify organizational patterns from business reality content.
        Focus on communication patterns, power dynamics, tensions.
        """
        
        payload = {
            "business_reality_content": business_reality_md[:8000],
            "business_context": (context_md or "")[:2000]
        }
        
        try:
            dynamics = chat(
                std_model(),
                [
                    {"role": "system", "content": ORGANIZATIONAL_DYNAMICS_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            return dynamics or "# Layer 2 — Organizational Dynamics\n(No patterns identified)\n"
        except Exception as e:
            logger.error(f"OrganizationalDynamicsAgent error: {e}")
            return "# Layer 2 — Organizational Dynamics\n(Error analyzing dynamics)\n"
