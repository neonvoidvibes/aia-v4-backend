import os
import logging
from typing import List, Dict, Any, Tuple

from utils.prompts import ENRICHMENT_PROMPT_TEMPLATE
from utils.llm_api_utils import _call_gemini_non_stream_with_retry, CircuitBreakerOpen

logger = logging.getLogger(__name__)

def _format_chat_for_enrichment(messages: List[Dict[str, Any]]) -> str:
    """Formats a list of chat messages into a simple string log."""
    log_lines = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "").strip()
        if content:
            log_lines.append(f"{role}: {content}")
    return "\n".join(log_lines)

def _extract_summary_from_enriched_log(log: str) -> str:
    """Extracts the summary from the YAML frontmatter of the enriched log using regex."""
    try:
        # Regex to find the YAML frontmatter block and capture the summary field
        match = re.search(r'---\s*\n(.*?)\n---', log, re.DOTALL)
        if match:
            frontmatter = match.group(1)
            summary_match = re.search(r'summary:\s*(?:"(.*?)"|\'(.*?)\'|(.*))', frontmatter)
            if summary_match:
                # The regex captures content in quotes or unquoted content.
                # One of the groups will have the summary.
                summary = next((s for s in summary_match.groups() if s is not None), None)
                if summary:
                    return summary.strip()
    except Exception as e:
        logger.warning(f"Could not extract summary from enriched log frontmatter using regex: {e}")

    # Fallback to first meaningful line if regex fails or no summary is found
    fallback_summary = next((line for line in log.splitlines() if line.strip() and not line.strip() == '---'), "No summary available.")
    return fallback_summary


def enrich_chat_log(messages: List[Dict[str, Any]], google_api_key: str) -> Tuple[str, str]:
    """
    Takes a chat history and enriches it using Gemini Flash into a structured log.

    Args:
        messages: The list of chat message dictionaries.
        google_api_key: The API key for Google Gemini.

    Returns:
        A tuple containing (structured_content, summary).
    """
    logger.info(f"Enriching chat log with {len(messages)} messages.")
    if not messages:
        return "", "Empty chat session."

    chat_log_string = _format_chat_for_enrichment(messages)
    
    prompt = ENRICHMENT_PROMPT_TEMPLATE.format(chat_log_string=chat_log_string)

    try:
        # Use Gemini Flash for speed and cost-effectiveness
        structured_content = _call_gemini_non_stream_with_retry(
            model_name="gemini-1.5-flash-latest", # Using latest flash model
            max_tokens=8192, # Generous token limit for the output
            system_instruction="You are an AI assistant that processes chat logs into structured Markdown documents according to user-provided templates.",
            messages=[{"role": "user", "content": prompt}],
            api_key=google_api_key,
            temperature=0.1 # Low temperature for consistent formatting
        )
        
        if not structured_content:
            logger.error("Log enrichment failed: Gemini returned an empty response.")
            return chat_log_string, "Failed to enrich log."

        summary = _extract_summary_from_enriched_log(structured_content)
        logger.info(f"Successfully enriched chat log. Summary: '{summary}'")
        return structured_content, summary

    except CircuitBreakerOpen as e:
        logger.error(f"Log enrichment failed: {e}")
        return chat_log_string, "Enrichment service unavailable."
    except Exception as e:
        logger.error(f"An unexpected error occurred during log enrichment: {e}", exc_info=True)
        return chat_log_string, "An unexpected error occurred during enrichment."