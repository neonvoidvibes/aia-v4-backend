import json
import logging
from typing import List, Dict, Any

from .base import Agent
from ._llm import chat
from .prompts import CONTEXT_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class ContextAgent(Agent):
    name = "context"

    def run(self, segments: List[Dict[str, Any]], repetition_analysis: Dict[str, Any] = None) -> str:
        """Extract business context to set clear context for other agents.
        Returns markdown only. Focus on business purpose, stakeholders, constraints.
        """
        # Combine all segments for context analysis
        combined_text = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')}"
            for s in segments
        ])
        
        payload = {
            "segments": segments,
            "combined_text": combined_text[:8000],  # Limit to avoid token issues
            "language_hint": "auto",  # Let the LLM detect language
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            context = chat(
                std_model(),
                [
                    {"role": "system", "content": CONTEXT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=1200,
                temperature=0.3,
            )
            return context or "# Business Context\n(No context extracted)\n"
        except Exception as e:
            logger.error(f"ContextAgent error: {e}")
            return "# Business Context\n(Error extracting context)\n"

    def refine(self, segments: List[Dict[str, Any]], previous_output: str, feedback: str, repetition_analysis: Dict[str, Any] = None) -> str:
        """Refine previous context analysis based on reality check feedback."""
        
        # Combine segments for reference
        combined_text = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')}"
            for s in segments
        ])
        
        payload = {
            "original_transcript": combined_text[:6000],
            "previous_context_analysis": previous_output[:3000],
            "reality_check_feedback": feedback[:2000],
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            from .prompts import CONTEXT_REFINEMENT_SYS
            
            refined_context = chat(
                std_model(),
                [
                    {"role": "system", "content": CONTEXT_REFINEMENT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=1400,
                temperature=0.3,
            )
            return refined_context or previous_output  # Fallback to previous if refinement fails
        except Exception as e:
            logger.error(f"ContextAgent refinement error: {e}")
            return previous_output

