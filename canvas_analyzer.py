import os
import json
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI, OpenAIError # Import OpenAIError for specific exception handling
from dotenv import load_dotenv # Add dotenv for standalone testing

logger = logging.getLogger(__name__)

# Global variable for the client, initialized later
openai_client: Optional[OpenAI] = None

def initialize_openai_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Initializes and returns the OpenAI client."""
    global openai_client
    if openai_client:
        return openai_client

    try:
        key_to_use = api_key or os.getenv("OPENAI_API_KEY")
        if not key_to_use:
            logger.error("CanvasAnalyzer: OPENAI_API_KEY not found in environment and not provided directly.")
            raise OpenAIError("OPENAI_API_KEY is not set.")
        
        openai_client = OpenAI(api_key=key_to_use)
        logger.info("CanvasAnalyzer: OpenAI client initialized successfully.")
        return openai_client
    except OpenAIError as e: # Catch specific OpenAIError first
        logger.error(f"CanvasAnalyzer: OpenAI API Error during client initialization: {e}")
        openai_client = None
        return None
    except Exception as e:
        logger.error(f"CanvasAnalyzer: Generic error during OpenAI client initialization: {e}", exc_info=True)
        openai_client = None
        return None

# System prompt for the Canvas Analysis Agent
CANVAS_SYSTEM_PROMPT_TEMPLATE = """
You are an analytical AI. Your primary task is to analyze the provided **Transcript Segment** through three distinct perspectives: Mirror, Lens, and Portal.
Use the **Static Documents** only as supplementary background or reference if directly relevant to understanding the transcript content. The core of your analysis should stem from the live conversation in the transcript.

When formulating highlights, aim for a tone that reflects collective intelligence (e.g., using "we" or phrasing that implies shared understanding or discussion points where appropriate and natural), but avoid rigidly prefixing every highlight with "we are...". The language should be adaptive and sound natural.

For each perspective (Mirror, Lens, Portal), identify 3-5 key highlights. Each highlight must be a descriptive short sentence or phrase. Also, provide a concise 'explanation' (1-2 sentences) elaborating slightly on each highlight.

Consolidate similar insights within each category. The goal is to surface salient, actionable, or thought-provoking points directly from the transcript.

The current analysis is focused on the time window: {time_window_label}.

**Primary Source for Analysis:**
<transcript_segment>
{transcript_segment}
</transcript_segment>

**Supplementary Static Documents (Use only for context if needed to understand the transcript):**
<static_documents>
{static_docs_content}
</static_documents>

Output *only* a single, valid JSON object with three top-level keys: "mirror", "lens", and "portal".
Each key should map to an array of objects. Each object in these arrays must have two string keys: "highlight" (the descriptive sentence/phrase) and "explanation".

Example of a single highlight object:
{{"highlight": "Exploring new marketing strategies is a current focus.", "explanation": "This refers to the discussion points around diversifying marketing efforts to reach new audiences, as heard in the transcript."}}

Ensure your entire response is a single JSON object.
"""

def analyze_transcript_for_canvas(
    transcript_segment: str,
    static_docs_content: str,
    time_window_label: str,
    agent_name: str, 
    event_id: Optional[str] 
) -> Dict[str, List[Dict[str, str]]]:
    """
    Analyzes transcript segment and static documents using GPT-4.1-mini.
    """
    client = initialize_openai_client() # Ensure client is initialized
    if not client:
        logger.error("CanvasAnalyzer: OpenAI client not available for analysis.")
        return {"mirror": [], "lens": [], "portal": []}

    max_transcript_len = 30000  
    max_static_docs_len = 15000 

    if len(transcript_segment) > max_transcript_len:
        logger.warning(f"CanvasAnalyzer: Transcript segment length ({len(transcript_segment)}) exceeds max ({max_transcript_len}). Truncating.")
        transcript_segment = transcript_segment[-max_transcript_len:] 
    
    if len(static_docs_content) > max_static_docs_len:
        logger.warning(f"CanvasAnalyzer: Static docs content length ({len(static_docs_content)}) exceeds max ({max_static_docs_len}). Truncating.")
        static_docs_content = static_docs_content[:max_static_docs_len]

    prompt_content = CANVAS_SYSTEM_PROMPT_TEMPLATE.format(
        time_window_label=time_window_label,
        static_docs_content=static_docs_content if static_docs_content else "No static documents provided for this analysis.",
        transcript_segment=transcript_segment if transcript_segment else "No transcript segment provided for this analysis."
    )

    logger.info(f"CanvasAnalyzer: Sending analysis request to OpenAI for agent '{agent_name}', event '{event_id}', window '{time_window_label}'. Prompt length (approx): {len(prompt_content)} chars.")
    
    try:
        # Using "gpt-4.1-mini" as requested. This model ID is valid and accessible via API endpoint.
        model_to_use = os.getenv("CANVAS_ANALYSIS_MODEL", "gpt-4.1-mini") 
        logger.info(f"CanvasAnalyzer: Using model: {model_to_use}")

        completion = client.chat.completions.create(
            model=model_to_use, 
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing meeting transcripts and extracting insights based on provided perspectives. Your output must be a single, valid JSON object as per the user's detailed instructions. Focus analysis on the transcript_segment."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3,
            response_format={"type": "json_object"} # Request JSON mode
        )

        response_text = completion.choices[0].message.content
        logger.debug(f"CanvasAnalyzer: Raw response from OpenAI: {response_text[:500]}...")

        if not response_text:
            logger.error("CanvasAnalyzer: Received empty response from OpenAI.")
            return {"mirror": [], "lens": [], "portal": []}
        
        insights = json.loads(response_text) # No need to strip ```json anymore with response_format

        if not all(key in insights for key in ["mirror", "lens", "portal"]):
            logger.error(f"CanvasAnalyzer: OpenAI response missing one or more required keys (mirror, lens, portal). Response: {insights}")
            return {"mirror": [], "lens": [], "portal": []}
        
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
        logger.error(f"CanvasAnalyzer: Failed to decode JSON response from OpenAI: {e}. Response was: {response_text[:1000] if 'response_text' in locals() else 'N/A'}", exc_info=True)
        return {"mirror": [], "lens": [], "portal": []}
    except OpenAIError as e: # Catch specific OpenAI errors
        logger.error(f"CanvasAnalyzer: OpenAI API Error: {e}", exc_info=True)
        return {"mirror": [], "lens": [], "portal": []}
    except Exception as e:
        logger.error(f"CanvasAnalyzer: Error during OpenAI API call or processing: {e}", exc_info=True)
        return {"mirror": [], "lens": [], "portal": []}

if __name__ == '__main__':
    load_dotenv() # Load .env for standalone testing
    # Basic logging setup for standalone testing
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("Canvas Analyzer Standalone Test Started")
    
    # Attempt to initialize client immediately for test
    client_for_test = initialize_openai_client()
    if not client_for_test:
        logger.critical("Failed to initialize OpenAI client for standalone test. Ensure OPENAI_API_KEY is set in .env")
    else:
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

        if not any(cat_insights for cat_insights in test_insights.values() if isinstance(cat_insights, list) and len(cat_insights) > 0):
            print("\nWARNING: Test output was empty or malformed. Check logs for errors and ensure your API key is valid and has quota.")
    logger.info("Canvas Analyzer Standalone Test Finished")