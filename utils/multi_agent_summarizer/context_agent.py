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

    def run(self, segments: List[Dict[str, Any]]) -> str:
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
            "language_hint": "auto"  # Let the LLM detect language
        }
        
        try:
            context = chat(
                std_model(),
                [
                    {"role": "system", "content": CONTEXT_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=1200,
                temperature=0.1,
            )
            return context or "# Business Context\n(No context extracted)\n"
        except Exception as e:
            logger.error(f"ContextAgent error: {e}")
            return "# Business Context\n(Error extracting context)\n"

