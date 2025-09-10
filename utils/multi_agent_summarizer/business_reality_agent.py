import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat
from .prompts import BUSINESS_REALITY_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class BusinessRealityAgent(Agent):
    name = "business_reality"

    def run(self, segments: List[Dict[str, Any]], context_md: str | None = None) -> str:
        """Extract explicit business content: decisions, tasks, commitments, constraints.
        Returns markdown focusing on concrete business realities discussed.
        """
        # Combine segments for analysis
        combined_text = ""
        for s in segments:
            segment_text = f"\n--- Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min) ---\n"
            segment_text += s.get('text', '')
            combined_text += segment_text
        
        # Calculate basic meeting metadata
        seg_min = min([s.get("start_min", 0) for s in segments] or [0])
        seg_max = max([s.get("end_min", 0) for s in segments] or [0])
        duration = max(1, int(seg_max - seg_min))

        payload = {
            "combined_text": combined_text[:12000],  # Limit for token management
            "context_excerpt": (context_md or "")[:2000],
            "meeting_duration": duration,
            "segment_count": len(segments)
        }
        
        try:
            business_reality = chat(
                std_model(),
                [
                    {"role": "system", "content": BUSINESS_REALITY_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2500,
                temperature=0.1,
            )
            return business_reality or "# Layer 1 — Business Reality\n(No business content extracted)\n"
        except Exception as e:
            logger.error(f"BusinessRealityAgent error: {e}")
            return "# Layer 1 — Business Reality\n(Error extracting business content)\n"

    def refine(self, segments: List[Dict[str, Any]], previous_output: str, feedback: str, context_md: str | None = None) -> str:
        """Refine previous business reality analysis based on reality check feedback."""
        
        # Combine segments for reference
        combined_text = ""
        for s in segments:
            segment_text = f"\n--- Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min) ---\n"
            segment_text += s.get('text', '')
            combined_text += segment_text
        
        payload = {
            "original_transcript": combined_text[:8000],
            "previous_business_reality": previous_output[:4000],
            "reality_check_feedback": feedback[:3000],
            "context_excerpt": (context_md or "")[:1500]
        }
        
        try:
            from .prompts import BUSINESS_REALITY_REFINEMENT_SYS
            
            refined_reality = chat(
                std_model(),
                [
                    {"role": "system", "content": BUSINESS_REALITY_REFINEMENT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2800,
                temperature=0.1,
            )
            return refined_reality or previous_output
        except Exception as e:
            logger.error(f"BusinessRealityAgent refinement error: {e}")
            return previous_output
