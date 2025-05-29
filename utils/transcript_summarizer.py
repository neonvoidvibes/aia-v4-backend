import os
import json
import logging
import re # Added missing import
from typing import Optional, Dict, Any
from anthropic import Anthropic, APIError
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """
You are an expert meeting summarization and analysis AI. Your primary objective is to transform the provided meeting transcript into a structured, dense JSON object. This JSON will serve as a rich memory artifact, enabling future AI interactions to have a deep and accurate understanding of past discussions.

**Core Principles:**
1.  **Accuracy and Grounding:** All information in the JSON output MUST be directly derived from the provided transcript. DO NOT invent, infer beyond clear implications, or hallucinate information. If specific details for a field are not present, use `null`, an empty array `[]`, or an empty string `""` as appropriate for the field's type, or omit optional fields.
2.  **Objectivity:** Focus on extracting factual information, stated intentions, and clear discussion points.
3.  **Structured Output:** Your entire response MUST be a single, valid JSON object. No explanatory text, greetings, or apologies outside of the JSON structure.
4.  **Conciseness within Detail:** Be concise in your descriptions, but ensure all critical elements are captured.
5.  **PII Awareness:** The transcript may have PII placeholders (e.g., `[PERSON_REDACTED]`). Preserve these placeholders as-is in your summary; do not attempt to fill them or guess the original PII.

**Input Context (Placeholders will be filled by the calling system):**
*   `original_filename`: {original_filename}
*   `source_s3_key`: {source_s3_key}
*   `agent_name`: {agent_name}
*   `event_id`: {event_id}
*   `current_utc_timestamp`: {current_utc_timestamp}

**JSON Output Schema:**

The JSON object MUST adhere to the following structure:

{{
  "metadata": {{
    "original_filename": "{original_filename}",
    "source_s3_key": "{source_s3_key}",
    "agent_name": "{agent_name}",
    "event_id": "{event_id}",
    "summarization_timestamp_utc": "{current_utc_timestamp}",
    "transcript_language_code": "string", // ISO 639-1 code (e.g., "en", "sv"). Infer from transcript.
    "estimated_duration_minutes": "integer | null" // Optional: Estimate meeting duration if discernible from timestamps.
  }},
  "overall_summary": "string", // A concise (2-4 sentences) executive summary of the entire meeting.
  "key_discussion_points": [ // Array of strings. Key distinct topics, themes, or significant points made. Aim for 5-10.
    "string"
  ],
  "action_items": [ // Array of objects.
    {{
      "task_description": "string", // Clear description of the action.
      "assigned_to": ["string | null"], // List of names/roles, or ["Unassigned"] or null.
      "due_date": "string | null", // (e.g., "YYYY-MM-DD") or null.
      "status": "string | null", // (e.g., "Open", "In Progress", "Completed") - If mentioned. Default to "Open".
      "notes_context": "string | null" // Brief supporting context from transcript.
    }}
  ],
  "decisions_made": [ // Array of objects.
    {{
      "decision_description": "string", // Clear description of the decision.
      "decision_maker": ["string | null"], // Who made or confirmed the decision, if clear.
      "supporting_reasons": "string | null", // Brief context or reasons.
      "timestamp_reference": "string | null" // E.g., "Towards the end of the meeting" or a specific timestamp if very clear.
    }}
  ],
  "questions_unanswered": [ // Array of objects. Important questions raised that were NOT definitively answered in THIS transcript.
    {{
      "question_text": "string",
      "raised_by": "string | null",
      "context": "string | null" // Context of when/why it was asked.
    }}
  ],
  "key_entities_mentioned": {{ // Object. Extract key named entities.
    "people": ["string"], // Unique names of individuals clearly mentioned.
    "organizations_clients": ["string"], // Unique names of companies, clients, or external organizations.
    "projects_initiatives": ["string"], // Unique names of specific projects, products, or initiatives.
    "key_terms_glossary": [ // Array of objects for domain-specific terms or acronyms defined or heavily discussed.
        {{
            "term": "string",
            "definition_or_context": "string | null" // If defined or explained.
        }}
    ]
  }},
  "sentiment_and_tone": {{ // Object. Overall qualitative assessment.
    "dominant_sentiment": "string | null", // e.g., "Positive", "Negative", "Neutral", "Mixed", "Constructive", "Contentious".
    "key_sentiment_indicators": ["string | null"] // List of 2-3 phrases or topics from transcript that strongly indicate the sentiment.
  }},
  "potential_risks_or_challenges": [ // Array of strings. Explicitly mentioned risks or challenges.
    "string"
  ],
  "opportunities_or_proposals": [ // Array of strings. Explicitly mentioned opportunities, new ideas, or proposals.
    "string"
  ],
  "meeting_outcomes_or_next_steps_summary": "string | null" // A brief summary of stated outcomes or agreed next steps beyond specific action items.
}}

**Instructions for Content Generation:**
1.  **Fill Placeholders Programmatically:** The placeholders like `{{original_filename}}` in the `metadata` section of the JSON schema will be filled by the system calling you. Your role is to generate the content for all other fields based on the transcript.
2.  **Infer Language and Duration:** Determine `metadata.transcript_language_code` (e.g., "sv", "en") and, if possible, `metadata.estimated_duration_minutes` from the transcript.
3.  **Adherence to Schema:** Strictly follow the JSON schema. Ensure all specified keys are present where applicable, and use correct data types (`string`, `array`, `object`, `null`, `integer`).
4.  **Conciseness & Accuracy:** Be concise yet comprehensive. All information must be grounded in the provided `transcript_content`.
5.  **Handle Missing Info:** For optional fields or when data isn't in the transcript, use `null` for singular string/integer fields, or an empty array `[]` for array fields. For `action_items.assigned_to`, use `["Unassigned"]` or `null` if no one is specified.
6.  **Filter Filler Phrases:** The transcript may contain repetitive conversational filler phrases (e.g., "sättet att tänka," "nu borde hon expandera i," "nu bombexploderar jag i," "liksom," "alltså så här"). Your summarization should focus on the substantive content and filter out these verbal tics when extracting meaningful information. Do not include these fillers in the summary fields.
7.  **Focus on Meaning:** Extract semantic meaning. Do not just copy-paste long sentences unless it is for providing very specific, short `context` in fields like `action_items` or `decisions_made`.
8.  **Clarity:** Ensure the output is easily understandable by another AI or a human reviewing this memory.

**Transcript Content to Process:**
```
{transcript_content}
```

Your entire output MUST be a single, valid JSON object as described above.
"""

def generate_transcript_summary(
    transcript_content: str,
    original_filename: str,
    agent_name: str,
    event_id: str,
    source_s3_key: str,
    llm_client: Anthropic,
    model_name: Optional[str] = None, # Allows override if passed
    max_tokens: int = 4000
) -> Optional[Dict[str, Any]]:
    """
    Generates a structured JSON summary of a transcript using an LLM.
    """
    if not transcript_content:
        logger.warning("generate_transcript_summary: No transcript content provided.")
        return None

    current_utc_timestamp = datetime.now(timezone.utc).isoformat()

    prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
        transcript_content=transcript_content,
        original_filename=original_filename,
        source_s3_key=source_s3_key,
        agent_name=agent_name,
        event_id=event_id,
        current_utc_timestamp=current_utc_timestamp
    )

    # Use explicitly passed model_name, then SUMMARY_LLM_MODEL_NAME, then the requested default
    final_model_name = model_name or os.getenv("SUMMARY_LLM_MODEL_NAME", "claude-sonnet-4-20250514")

    try:
        logger.info(f"Generating summary for '{original_filename}' using model '{final_model_name}'. Agent: {agent_name}, Event: {event_id}. Prompt length (approx): {len(prompt_content)}")
        # Note: We use the system parameter for the main instructions here,
        # and the user message is just the transcript content itself, prefixed by the context.
        # This aligns better with how Claude models are often used for structured data extraction.
        
        # However, for this specific prompt, the instruction to the LLM is to act as a summarizer
        # and the transcript is part of the overall "instructions" (user message).
        # The actual "system" part of the Claude API call can be minimal or reinforce its role.

        response = llm_client.messages.create(
            model=final_model_name,
            max_tokens=max_tokens,
            system="You are an AI assistant specialized in analyzing meeting transcripts and extracting structured insights into JSON format. Follow the user's instructions precisely and output ONLY valid JSON.",
            messages=[
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.2, # Lower temperature for more deterministic JSON output
        )

        if not response.content or not isinstance(response.content, list) or len(response.content) == 0:
            logger.error("LLM response was empty or not in expected format.")
            return None

        raw_llm_output = response.content[0].text
        logger.debug(f"Raw LLM output for summary (first 500 chars): {raw_llm_output[:500]}...")

        # Attempt to parse the JSON
        # LLMs sometimes wrap JSON in markdown code blocks (```json ... ```)
        # or might have leading/trailing text despite instructions.
        
        cleaned_llm_output = raw_llm_output.strip()
        if cleaned_llm_output.startswith("```json"):
            cleaned_llm_output = cleaned_llm_output[len("```json"):]
        if cleaned_llm_output.startswith("```"): # Catch if it's just ```
            cleaned_llm_output = cleaned_llm_output[len("```"):]
        if cleaned_llm_output.endswith("```"):
            cleaned_llm_output = cleaned_llm_output[:-len("```")]
        
        cleaned_llm_output = cleaned_llm_output.strip()

        try:
            summary_data = json.loads(cleaned_llm_output)
            logger.info("Successfully parsed JSON directly after cleaning.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode cleaned JSON from LLM response: {e}. Cleaned response sample: {cleaned_llm_output[:500]}...")
            # If direct parsing fails, try the regex again as a more specific fallback for ```json { ... } ```
            match = re.search(r"(\{.*?\})", cleaned_llm_output, re.DOTALL) # More general regex to find first { to last }
            if match:
                logger.info("Attempting to parse extracted JSON block using regex.")
                try:
                    summary_data = json.loads(match.group(1))
                    logger.info("Successfully parsed JSON from regex-extracted block.")
                except json.JSONDecodeError as e_nested:
                    logger.error(f"Failed to decode JSON from regex-extracted block: {e_nested}. Block sample: {match.group(1)[:500]}...")
                    return None
            else:
                logger.error("No JSON object found in LLM output even with regex.")
                return None
        
        # Basic validation of the summary structure
        if not isinstance(summary_data, dict) or "metadata" not in summary_data or "overall_summary" not in summary_data:
            logger.error(f"Parsed JSON does not match expected top-level structure. Parsed: {str(summary_data)[:500]}")
            return None

        # Inject the source_s3_key and other crucial metadata again, overriding if LLM tried to fill them.
        # This ensures system-provided values are authoritative.
        summary_data["metadata"]["original_filename"] = original_filename
        summary_data["metadata"]["source_s3_key"] = source_s3_key
        summary_data["metadata"]["agent_name"] = agent_name
        summary_data["metadata"]["event_id"] = event_id
        summary_data["metadata"]["summarization_timestamp_utc"] = current_utc_timestamp


        logger.info(f"Successfully generated and parsed summary for '{original_filename}'.")
        return summary_data

    except APIError as e:
        logger.error(f"Anthropic APIError generating summary for '{original_filename}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating summary for '{original_filename}': {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # This basic test requires ANTHROPIC_API_KEY to be set in the environment
    # and the anthropic library installed.
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)

    api_key_test = os.getenv("ANTHROPIC_API_KEY")
    if not api_key_test:
        logger.error("ANTHROPIC_API_KEY not found. Cannot run test.")
    else:
        test_client = Anthropic(api_key=api_key_test)
        sample_transcript_content = """
[10:00:00 - 10:00:15 UTC] Alice: Good morning team. Today's focus is project Alpha. We need to finalize the Q3 roadmap.
[10:00:15 - 10:00:30 UTC] Bob: Morning. I think the key deadline for module X is too tight. [PERSON_REDACTED] from sales also mentioned this.
[10:00:30 - 10:00:45 UTC] Carol: I agree. We need to review the resource allocation. My email is carol@example.com.
[10:00:45 - 10:01:00 UTC] Alice: Okay, action item for Bob: review module X deadline. Let's also discuss the new marketing campaign proposed by Beta Corp.
[10:01:00 - 10:01:15 UTC] Bob: Decision: We will push the module X deadline by one week.
[10:01:15 - 10:01:30 UTC] Carol: What about the budget for the Beta Corp campaign? That's an unanswered question.
        """
        summary = generate_transcript_summary(
            transcript_content=sample_transcript_content,
            original_filename="meeting_transcript_q3_alpha.txt",
            agent_name="test_agent",
            event_id="event_123",
            source_s3_key="path/to/meeting_transcript_q3_alpha.txt",
            llm_client=test_client
        )
        if summary:
            print("\n--- Generated Summary JSON ---")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("--- End of Summary ---")
        else:
            print("\n--- Summary Generation Failed ---")