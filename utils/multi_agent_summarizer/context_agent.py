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

    def run(self, segments: List[Dict[str, Any]], repetition_analysis: Dict[str, Any] = None, meeting_datetime: str = None) -> str:
        """Extract business context to set clear context for other agents.
        Returns markdown only. Focus on business purpose, stakeholders, constraints.
        """
        # Validate segments have content
        if not segments or not any(s.get('text', '').strip() for s in segments):
            logger.warning(f"ContextAgent received empty or invalid segments: {len(segments) if segments else 0} segments")
            return "# Business Context\n(No content in segments to analyze)\n"
        
        # Combine all segments for context analysis
        combined_text = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')}"
            for s in segments if s.get('text', '').strip()  # Only include segments with actual text
        ])
        
        # Additional validation after combining
        if not combined_text.strip():
            logger.warning("ContextAgent: combined_text is empty after filtering segments")
            return "# Business Context\n(No valid text content found in segments after filtering)\n"
        
        payload = {
            "segments": segments,
            "combined_text": combined_text[:8000],  # Limit to avoid token issues
            "language_hint": "auto",  # Let the LLM detect language
            "repetition_analysis": repetition_analysis or {"exclusion_instructions": "No repetitive phrases detected."}
        }
        
        try:
            logger.debug(f"ContextAgent requesting analysis for {len(segments)} segments with combined_text length: {len(combined_text)}")
            context = chat(
                std_model(),
                [
                    {"role": "system", "content": CONTEXT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=1200,
                temperature=0.2,  # Balanced temperature for content quality and instruction following
            )
            
            if not context or context.strip() == "":
                logger.warning(f"ContextAgent received empty response. Payload keys: {list(payload.keys())}, segments count: {len(segments)}")
                # Try to extract any business content manually as fallback
                if any(seg.get('text', '').strip() for seg in segments):
                    return "# Business Context\n\n- **Meeting content**: Business discussion detected but context extraction failed - manual review recommended\n"
                else:
                    return "# Business Context\n(No context extracted - empty or invalid segments)\n"
            
            # Prepend datetime if available
            if meeting_datetime and context:
                datetime_header = f"**Meeting Date/Time:** {meeting_datetime}\n\n"
                if context.startswith("# Business Context"):
                    # Insert after the header
                    lines = context.split('\n', 1)
                    if len(lines) == 2:
                        context = f"{lines[0]}\n\n{datetime_header}{lines[1]}"
                    else:
                        context = f"{lines[0]}\n\n{datetime_header}"
                else:
                    context = f"{datetime_header}{context}"
            
            return context
        except Exception as e:
            logger.error(f"ContextAgent error: {e}")
            return "# Business Context\n(Error extracting context)\n"

    def refine(self, segments: List[Dict[str, Any]], previous_output: str, feedback: str, repetition_analysis: Dict[str, Any] = None, meeting_datetime: str = None) -> str:
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
            output_content = refined_context or previous_output  # Fallback to previous if refinement fails
            
            # Prepend datetime if available
            if meeting_datetime and output_content:
                datetime_header = f"**Meeting Date/Time:** {meeting_datetime}\n\n"
                if output_content.startswith("# Business Context"):
                    # Insert after the header
                    lines = output_content.split('\n', 1)
                    if len(lines) == 2:
                        output_content = f"{lines[0]}\n\n{datetime_header}{lines[1]}"
                    else:
                        output_content = f"{lines[0]}\n\n{datetime_header}"
                else:
                    output_content = f"{datetime_header}{output_content}"
            
            return output_content
        except Exception as e:
            logger.error(f"ContextAgent refinement error: {e}")
            return previous_output

