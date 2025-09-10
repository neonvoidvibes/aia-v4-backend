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

    def run(self, segments: List[Dict[str, Any]], context_md: str | None = None, repetition_analysis: Dict[str, Any] = None, meeting_datetime: str = None) -> str:
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
            "segment_count": len(segments),
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            logger.debug(f"BusinessRealityAgent requesting analysis for {len(segments)} segments, combined_text length: {len(combined_text)}")
            business_reality = chat(
                std_model(),
                [
                    {"role": "system", "content": BUSINESS_REALITY_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2500,
                temperature=0.1,  # Lower temperature for better instruction following
            )
            
            if not business_reality or business_reality.strip() == "":
                logger.warning(f"BusinessRealityAgent received empty response. Payload keys: {list(payload.keys())}, segments count: {len(segments)}")
                return "# Layer 1 — Business Reality\n(No business content extracted - empty API response)\n"
            
            # Prepend datetime if available
            if meeting_datetime and business_reality:
                datetime_header = f"**Meeting Date/Time:** {meeting_datetime}\n\n"
                if business_reality.startswith("# Layer 1 — Business Reality"):
                    # Insert after the header
                    lines = business_reality.split('\n', 1)
                    if len(lines) == 2:
                        business_reality = f"{lines[0]}\n\n{datetime_header}{lines[1]}"
                    else:
                        business_reality = f"{lines[0]}\n\n{datetime_header}"
                else:
                    business_reality = f"{datetime_header}{business_reality}"
            
            return business_reality
        except Exception as e:
            logger.error(f"BusinessRealityAgent error: {e}")
            return "# Layer 1 — Business Reality\n(Error extracting business content)\n"

    def refine(self, segments: List[Dict[str, Any]], previous_output: str, feedback: str, context_md: str | None = None, repetition_analysis: Dict[str, Any] = None, meeting_datetime: str = None) -> str:
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
            "context_excerpt": (context_md or "")[:1500],
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
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
                temperature=0.1,  # Lower temperature for better instruction following
            )
            
            # Apply datetime prepending if available and refined_reality has content
            if meeting_datetime and refined_reality:
                datetime_header = f"**Meeting Date/Time:** {meeting_datetime}\n\n"
                if refined_reality.startswith("# Layer 1 — Business Reality"):
                    # Insert after the header
                    lines = refined_reality.split('\n', 1)
                    if len(lines) == 2:
                        refined_reality = f"{lines[0]}\n\n{datetime_header}{lines[1]}"
                    else:
                        refined_reality = f"{lines[0]}\n\n{datetime_header}"
                else:
                    refined_reality = f"{datetime_header}{refined_reality}"
            
            return refined_reality or previous_output
        except Exception as e:
            logger.error(f"BusinessRealityAgent refinement error: {e}")
            return previous_output
