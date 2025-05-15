import os
import json
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
# Ensure OPENAI_API_KEY is set in the environment
try:
    openai_client = OpenAI() # API key is read from OPENAI_API_KEY env var
    logger.info("CanvasAnalyzer: OpenAI client initialized.")
except Exception as e:
    logger.error(f"CanvasAnalyzer: Failed to initialize OpenAI client: {e}", exc_info=True)
    openai_client = None

# System prompt for the Canvas Analysis Agent
CANVAS_SYSTEM_PROMPT_TEMPLATE = """
You are an analytical AI. Your task is to analyze the provided meeting transcript segment and any accompanying static documents (like frameworks or organizational context) through three distinct perspectives: Mirror, Lens, and Portal.

Use "we" language (e.g., "we are discussing", "our focus seems to be") to reflect a sense of collective intelligence emerging from the conversation.

For each perspective, identify 3-5 key highlights. Each highlight should be a descriptive short sentence or phrase. Also, provide a concise 'explanation' (1-2 sentences) elaborating slightly on each highlight.

Consolidate similar insights within each category. The goal is to surface salient, actionable, or thought-provoking points.

The current analysis is focused on the time window: {time_window_label}.

Static documents provided for broader context:
<static_documents>
{static_docs_content}
</static_documents>

Transcript segment for analysis:
<transcript_segment>
{transcript_segment}
</transcript_segment>

Output *only* a single, valid JSON object with three top-level keys: "mirror", "lens", and "portal".
Each key should map to an array of objects. Each object in these arrays must have two string keys: "highlight" and "explanation".

Example of a single highlight object:
{{"highlight": "We are exploring new marketing strategies.", "explanation": "This refers to the discussion points around diversifying our marketing efforts to reach new audiences."}}

Ensure your entire response is a single JSON object.
"""

def analyze_transcript_for_canvas(
    transcript_segment: str,
    static_docs_content: str,
    time_window_label: str,
    agent_name: str, # Added for potential future use or more specific logging
    event_id: Optional[str] # Added for potential future use or more specific logging
) -> Dict[str, List[Dict[str, str]]]:
    """
    Analyzes transcript segment and static documents using GPT-4.1-mini.

    Args:
        transcript_segment: The segment of the transcript to analyze.
        static_docs_content: Combined content of static documents (frameworks, org context).
        time_window_label: User-friendly label for the time window being analyzed.
        agent_name: Name of the agent for context/logging.
        event_id: ID of the event for context/logging.

    Returns:
        A dictionary structured as:
        {
            "mirror": [{"highlight": "...", "explanation": "..."}],
            "lens": [{"highlight": "...", "explanation": "..."}],
            "portal": [{"highlight": "...", "explanation": "..."}]
        }
        Returns an empty structure on failure.
    """
    if not openai_client:
        logger.error("CanvasAnalyzer: OpenAI client not available. Cannot perform analysis.")
        return {"mirror": [], "lens": [], "portal": []}

    # Truncate inputs if they are excessively long to prevent very large prompts,
    # though GPT-4.1-mini has a large context window. This is a basic safeguard.
    # A more sophisticated approach would involve token counting.
    max_transcript_len = 30000  # Approx characters
    max_static_docs_len = 15000 # Approx characters

    if len(transcript_segment) > max_transcript_len:
        logger.warning(f"CanvasAnalyzer: Transcript segment length ({len(transcript_segment)}) exceeds max ({max_transcript_len}). Truncating.")
        transcript_segment = transcript_segment[-max_transcript_len:] # Keep the most recent part
    
    if len(static_docs_content) > max_static_docs_len:
        logger.warning(f"CanvasAnalyzer: Static docs content length ({len(static_docs_content)}) exceeds max ({max_static_docs_len}). Truncating.")
        static_docs_content = static_docs_content[:max_static_docs_len]


    prompt_content = CANVAS_SYSTEM_PROMPT_TEMPLATE.format(
        time_window_label=time_window_label,
        static_docs_content=static_docs_content if static_docs_content else "No static documents provided for this analysis.",
        transcript_segment=transcript_segment if transcript_segment else "No transcript segment provided for this analysis."
    )

    logger.info(f"CanvasAnalyzer: Sending analysis request to GPT-4.1-mini for agent '{agent_name}', event '{event_id}', window '{time_window_label}'. Prompt length (approx): {len(prompt_content)} chars.")
    
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4-0125-preview", # Using a placeholder, assuming gpt-4.1-mini is aliased or will be updated. For testing, use a known model. "gpt-4-turbo-preview" is good.
            # model="gpt-4-turbo-preview", # More common alias, or use the specific one if available like "gpt-4-vision-preview" if multimodal is intended later, or "gpt-4-1106-preview"
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing meeting transcripts and extracting insights based on provided perspectives. Your output must be a single, valid JSON object as per the user's detailed instructions."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3, # Lower temperature for more factual extraction
            # Ensure response_format is used correctly if available for the model version
            # For newer models, you might use: response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        logger.debug(f"CanvasAnalyzer: Raw response from GPT-4.1-mini: {response_text[:500]}...") # Log beginning of response

        if not response_text:
            logger.error("CanvasAnalyzer: Received empty response from GPT-4.1-mini.")
            return {"mirror": [], "lens": [], "portal": []}

        # Attempt to parse the JSON
        # The LLM might sometimes include markdown ```json ... ``` around the JSON.
        cleaned_response_text = response_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[len("```json"):]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-len("```")]
        cleaned_response_text = cleaned_response_text.strip()
        
        insights = json.loads(cleaned_response_text)

        # Validate basic structure
        if not all(key in insights for key in ["mirror", "lens", "portal"]):
            logger.error(f"CanvasAnalyzer: GPT-4.1-mini response missing one or more_required keys (mirror, lens, portal). Response: {insights}")
            return {"mirror": [], "lens": [], "portal": []}
        
        # Further ensure each category is a list of dicts with 'highlight' and 'explanation'
        for category in ["mirror", "lens", "portal"]:
            if not isinstance(insights[category], list):
                insights[category] = []
            insights[category] = [
                item for item in insights[category] 
                if isinstance(item, dict) and "highlight" in item and "explanation" in item
            ]

        logger.info(f"CanvasAnalyzer: Successfully parsed insights. Mirror: {len(insights['mirror'])}, Lens: {len(insights['lens'])}, Portal: {len(insights['portal'])} items.")
        return insights

    except json.JSONDecodeError as e:
        logger.error(f"CanvasAnalyzer: Failed to decode JSON response from GPT-4.1-mini: {e}. Response was: {response_text[:1000]}...", exc_info=True)
        return {"mirror": [], "lens": [], "portal": []}
    except Exception as e:
        logger.error(f"CanvasAnalyzer: Error during GPT-4.1-mini API call or processing: {e}", exc_info=True)
        return {"mirror": [], "lens": [], "portal": []}

if __name__ == '__main__':
    # Basic test (requires OPENAI_API_KEY to be set)
    logger.setLevel(logging.DEBUG)
    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    sample_transcript = """
    [10:00:00 - 10:00:15 UTC] Alice: Good morning, everyone. Let's kick off the Q4 budget discussion.
    [10:00:15 - 10:00:30 UTC] Bob: Morning. My main concern is our current team bandwidth. We're stretched thin.
    [10:00:30 - 10:00:45 UTC] Carol: I agree with Bob. While new projects are exciting, we need to ensure operational stability.
    [10:00:45 - 10:01:00 UTC] Alice: Valid points. We need to balance innovation with our capacity. Perhaps AI could help automate some of our manual reporting?
    [10:01:00 - 10:01:15 UTC] Bob: That's an interesting idea. Could free up significant time if feasible.
    """
    sample_static_docs = """
    Framework: Project Prioritization
    1. Strategic Alignment
    2. Resource Availability
    3. Potential ROI

    Org Context: We are currently in a phase of rapid growth and exploring new market segments.
    """
    
    if not openai_client:
        print("OpenAI client not initialized. Ensure OPENAI_API_KEY is set.")
    else:
        test_insights = analyze_transcript_for_canvas(
            transcript_segment=sample_transcript,
            static_docs_content=sample_static_docs,
            time_window_label="Last 1 minute",
            agent_name="test_agent",
            event_id="test_event"
        )
        print("\n--- Canvas Analyzer Test Output ---")
        print(json.dumps(test_insights, indent=2))
        print("--- End Test Output ---")

        if not any(test_insights.values()):
            print("\nWARNING: Test output was empty. Check logs for errors and ensure your API key is valid and has quota.")