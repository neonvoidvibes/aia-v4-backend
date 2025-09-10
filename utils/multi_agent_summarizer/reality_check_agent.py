import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat
from .prompts import REALITY_CHECK_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class RealityCheckAgent(Agent):
    name = "reality_check"

    def run(self, segments: List[Dict[str, Any]], context_md: str, business_reality_md: str, 
            organizational_dynamics_md: str, strategic_implications_md: str, next_actions_md: str) -> str:
        """Validate accuracy and usefulness of all previous layer outputs.
        Check against original transcript for accuracy and practical value.
        """
        
        # Create a sample of the original transcript for validation
        original_text_sample = "\n\n".join([
            f"Segment {s.get('id', '')} ({s.get('start_min', 0)}-{s.get('end_min', 0)} min): {s.get('text', '')[:500]}..."
            for s in segments[:3]  # Just first 3 segments to avoid token limits
        ])
        
        payload = {
            "original_transcript_sample": original_text_sample,
            "context_output": context_md[:2000],
            "business_reality_output": business_reality_md[:3000],
            "organizational_dynamics_output": organizational_dynamics_md[:2500],
            "strategic_implications_output": strategic_implications_md[:2500],
            "next_actions_output": next_actions_md[:2000]
        }
        
        try:
            reality_check = chat(
                std_model(),
                [
                    {"role": "system", "content": REALITY_CHECK_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=2500,
                temperature=0.1,
            )
            return reality_check or "# Reality Check Assessment\n(No assessment generated)\n"
        except Exception as e:
            logger.error(f"RealityCheckAgent error: {e}")
            return "# Reality Check Assessment\n(Error performing reality check)\n"