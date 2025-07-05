import os
import logging
import re
from typing import List, Dict, Any, Tuple

from utils.prompts import ENRICHMENT_PROMPT_TEMPLATE, ENRICHMENT_CONTINUATION_PROMPT_TEMPLATE
from utils.llm_api_utils import _call_gemini_non_stream_with_retry, CircuitBreakerOpen

logger = logging.getLogger(__name__)

# A simple heuristic to split the message list into chunks for the LLM
# This avoids creating a single massive prompt string.
# We aim for chunks of roughly 20 messages.
MESSAGES_PER_CHUNK = 20

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
        match = re.search(r'---\s*\n(.*?)\n---', log, re.DOTALL)
        if match:
            frontmatter = match.group(1)
            summary_match = re.search(r'summary:\s*(?:"(.*?)"|\'(.*?)\'|(.*))', frontmatter)
            if summary_match:
                summary = next((s for s in summary_match.groups() if s is not None), None)
                if summary:
                    return summary.strip()
    except Exception as e:
        logger.warning(f"Could not extract summary from enriched log frontmatter using regex: {e}")
    return "Summary could not be extracted."

def _count_turns_in_markdown(markdown_text: str) -> int:
    """Counts the number of '### Turn X' occurrences in the structured output."""
    return len(re.findall(r'^### Turn \d+', markdown_text, re.MULTILINE))

def enrich_chat_log(messages: List[Dict[str, Any]], google_api_key: str) -> Tuple[str, str]:
    """
    Takes a chat history and enriches it using a looping, chunked approach to ensure completeness.

    Args:
        messages: The list of chat message dictionaries.
        google_api_key: The API key for Google Gemini.

    Returns:
        A tuple containing (structured_content, summary).
    """
    logger.info(f"Starting enrichment for chat log with {len(messages)} messages.")
    if not messages:
        return "", "Empty chat session."

    unprocessed_messages = list(messages)
    full_structured_log = ""
    is_first_chunk = True
    max_loops = (len(messages) // MESSAGES_PER_CHUNK) + 5  # Safety break
    loop_count = 0

    while unprocessed_messages and loop_count < max_loops:
        loop_count += 1
        messages_in_chunk = unprocessed_messages[:MESSAGES_PER_CHUNK]
        chat_log_chunk_string = _format_chat_for_enrichment(messages_in_chunk)

        if is_first_chunk:
            prompt = ENRICHMENT_PROMPT_TEMPLATE.format(chat_log_string=chat_log_chunk_string)
        else:
            prompt = ENRICHMENT_CONTINUATION_PROMPT_TEMPLATE.format(
                previous_structured_log=full_structured_log,
                chat_log_string=chat_log_chunk_string
            )
        
        try:
            logger.info(f"Processing chunk of {len(messages_in_chunk)} messages (Loop {loop_count})...")
            partial_content = _call_gemini_non_stream_with_retry(
                model_name="gemini-2.5-flash",
                max_tokens=16384,
                system_instruction="You are an AI assistant that processes chat logs into structured Markdown documents according to user-provided templates.",
                messages=[{"role": "user", "content": prompt}],
                api_key=google_api_key,
                temperature=0.1
            )

            if not partial_content:
                logger.error(f"Log enrichment failed on loop {loop_count}: Gemini returned an empty response.")
                # If it fails, we stop and return what we have so far to avoid losing progress.
                break

            processed_turns = _count_turns_in_markdown(partial_content)
            logger.info(f"Chunk processed. Model returned {processed_turns} turns for {len(messages_in_chunk)} messages sent.")

            # Append the new content, handling the YAML header on the first run
            if is_first_chunk:
                full_structured_log = partial_content
                is_first_chunk = False
            else:
                # Strip any potential repeated headers from continuation calls
                partial_content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', partial_content, flags=re.DOTALL)
                full_structured_log += "\n\n" + partial_content

            # Update the list of unprocessed messages
            if processed_turns > 0 and processed_turns <= len(messages_in_chunk):
                unprocessed_messages = unprocessed_messages[processed_turns:]
            else:
                # If the model returns 0 turns or an unexpected number, stop to prevent infinite loops.
                logger.warning(f"Stopping enrichment loop due to unexpected turn count ({processed_turns}) from model.")
                break
        
        except CircuitBreakerOpen as e:
            logger.error(f"Log enrichment failed: {e}")
            break # Stop processing if the service is unavailable
        except Exception as e:
            logger.error(f"An unexpected error occurred during log enrichment loop: {e}", exc_info=True)
            break # Stop on other unexpected errors

    if loop_count >= max_loops:
        logger.error(f"Enrichment process exited due to exceeding max loop count ({max_loops}).")

    if not full_structured_log:
        logger.error("Enrichment resulted in an empty final document.")
        return _format_chat_for_enrichment(messages), "Failed to enrich log."

    summary = _extract_summary_from_enriched_log(full_structured_log)
    logger.info(f"Successfully enriched chat log. Final summary: '{summary}'")
    
    # Final cleanup to ensure no duplicate headers exist
    parts = full_structured_log.split('---')
    if len(parts) > 3: # More than one YAML block
        final_log = f"---\n{parts[1]}---\n" + "\n".join(p.strip() for p in parts[2:])
    else:
        final_log = full_structured_log

    return final_log, summary
