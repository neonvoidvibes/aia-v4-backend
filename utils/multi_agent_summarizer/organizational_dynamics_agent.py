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

    def run(self, segments: List[Dict[str, Any]], business_reality_md: str, context_md: str | None = None, repetition_analysis: Dict[str, Any] = None) -> str:
        """Identify organizational patterns from segments, business reality content.
        Focus on communication patterns, power dynamics, tensions.
        """
        
        # Combine segments for analysis
        combined_text = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')}"
            for s in segments
        ])
        
        payload = {
            "segments_content": combined_text[:10000],
            "business_reality_content": business_reality_md[:6000],
            "business_context": (context_md or "")[:2000],
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            dynamics = chat(
                std_model(),
                [
                    {"role": "system", "content": ORGANIZATIONAL_DYNAMICS_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            return dynamics or "# Layer 2 — Organizational Dynamics\n(No patterns identified)\n"
        except Exception as e:
            logger.error(f"OrganizationalDynamicsAgent error: {e}")
            return "# Layer 2 — Organizational Dynamics\n(Error analyzing dynamics)\n"

    def refine(self, segments: List[Dict[str, Any]], business_reality_md: str, previous_output: str, feedback: str, context_md: str | None = None, repetition_analysis: Dict[str, Any] = None) -> str:
        """Refine previous organizational dynamics analysis based on reality check feedback."""
        
        # Combine segments for reference
        combined_text = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')}"
            for s in segments
        ])
        
        payload = {
            "original_transcript": combined_text[:6000],
            "business_reality_content": business_reality_md[:4000],
            "previous_org_dynamics": previous_output[:3000],
            "reality_check_feedback": feedback[:2500],
            "business_context": (context_md or "")[:1500],
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            from .prompts import ORGANIZATIONAL_DYNAMICS_REFINEMENT_SYS
            
            refined_dynamics = chat(
                std_model(),
                [
                    {"role": "system", "content": ORGANIZATIONAL_DYNAMICS_REFINEMENT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2200,
                temperature=0.3,
            )
            return refined_dynamics or previous_output
        except Exception as e:
            logger.error(f"OrganizationalDynamicsAgent refinement error: {e}")
            return previous_output
