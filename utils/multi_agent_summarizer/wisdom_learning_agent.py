import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat
from .prompts import WISDOM_LEARNING_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class WisdomLearningAgent(Agent):
    name = "wisdom_learning"

    def run(self, segments: List[Dict[str, Any]], context_md: str, business_reality_md: str, 
            organizational_dynamics_md: str, strategic_implications_md: str, repetition_analysis: Dict[str, Any] = None) -> str:
        """Extract wisdom and learning insights using analytical frameworks from previous layers."""
        
        # Combine segments for reference
        original_text = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')}"
            for s in segments
        ])
        
        payload = {
            "original_segments": original_text[:15000],
            "context_content": context_md[:3000],
            "business_reality_content": business_reality_md[:4000],
            "organizational_dynamics_content": organizational_dynamics_md[:3000],
            "strategic_implications_content": strategic_implications_md[:3000],
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            wisdom_learning = chat(
                std_model(),
                [
                    {"role": "system", "content": WISDOM_LEARNING_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2500,
                temperature=0.3,
            )
            return wisdom_learning or "# Layer 5 — Wisdom and Learning\n(No wisdom insights generated)\n"
        except Exception as e:
            logger.error(f"WisdomLearningAgent error: {e}")
            return "# Layer 5 — Wisdom and Learning\n(Error generating wisdom insights)\n"

    def refine(self, segments: List[Dict[str, Any]], context_md: str, business_reality_md: str,
               organizational_dynamics_md: str, strategic_implications_md: str, previous_output: str, 
               feedback: str, repetition_analysis: Dict[str, Any] = None) -> str:
        """Refine previous wisdom and learning analysis based on reality check feedback."""
        
        # Combine segments for reference  
        original_text = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')}"
            for s in segments
        ])
        
        payload = {
            "original_segments": original_text[:15000],
            "context_content": context_md[:3000],
            "business_reality_content": business_reality_md[:4000],
            "organizational_dynamics_content": organizational_dynamics_md[:3000],
            "strategic_implications_content": strategic_implications_md[:3000],
            "previous_wisdom_analysis": previous_output[:4000],
            "reality_check_feedback": feedback[:2000],
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            from .prompts import WISDOM_LEARNING_REFINEMENT_SYS
            refined_wisdom = chat(
                std_model(),
                [
                    {"role": "system", "content": WISDOM_LEARNING_REFINEMENT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2500,
                temperature=0.3,
            )
            return refined_wisdom or previous_output
        except Exception as e:
            logger.error(f"WisdomLearningAgent refinement error: {e}")
            return previous_output