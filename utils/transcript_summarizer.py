import os
import json
import logging
from typing import Optional, Dict, Any
from anthropic import Anthropic, APIError
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _clean_json_string(json_str: str) -> str:
    """
    Clean basic JSON formatting issues from LLM output.
    """
    # Remove any leading/trailing whitespace
    json_str = json_str.strip()
    
    # Remove markdown code blocks
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):]
    elif json_str.startswith("```"):
        json_str = json_str[len("```"):]
    if json_str.endswith("```"):
        json_str = json_str[:-len("```")]
    
    return json_str.strip()

SYSTEM_PROMPT_TEMPLATE = """
## Core Mission
You are a sophisticated transcript analysis agent that creates comprehensive, chronologically-organized summaries optimized for AI agent memory and contextual understanding. Your summaries serve as working memory for conversational AI agents who need detailed context about past sessions.

## Critical Requirements

### Core Principles
1.  **Accuracy and Grounding:** All information in the JSON output MUST be directly derived from the provided transcript. DO NOT invent, infer beyond clear implications, or hallucinate information. If specific details for a field are not present, use `null`, an empty array `[]`, or an empty string `""` as appropriate for the field's type, or omit optional fields.
2.  **Objectivity:** Focus on extracting factual information, stated intentions, and clear discussion points.
3.  **Structured Output:** Your entire response MUST be a single, valid JSON object. No explanatory text, greetings, or apologies outside of the JSON structure.
4.  **Conciseness within Detail:** Be concise in your descriptions, but ensure all critical elements are captured.
5.  **PII Awareness:** The transcript may have PII placeholders (e.g., `[PERSON_REDACTED]`). Preserve these placeholders as-is in your summary; do not attempt to fill them or guess the original PII.

### 1. CHRONOLOGICAL ORGANIZATION IS MANDATORY
- **Always organize content by temporal sequence** - what happened first, second, third, etc.
- Include timeframe estimates for each major section
- Show how concepts built upon each other throughout the session
- Enable AI agents to reference "earlier in the session when..." or "building on what was discussed..."

### 2. SPECIFIC CONTENT OVER VAGUE DESCRIPTIONS
**NEVER write:** "Discussion of AI capabilities"
**ALWAYS write:** "AI capabilities: Pattern recognition in massive datasets, natural language interaction enabling human-like conversation, 24/7 availability for strategic consultation"

**NEVER write:** "Participants expressed concerns"  
**ALWAYS write:** "Concerns expressed: AI-generated text sounds too grammatically correct, employee privacy and consent requirements, maintaining authentic communication while using AI assistance"

### 3. MIRROR-LENS-PORTAL ANALYSIS REQUIRED
For each transcript, conduct thorough analysis across three levels:

**MIRROR (Explicit Content):**
- What was literally said, using participants' actual language
- Concrete themes identified from direct statements
- Surface-level patterns and topics covered

**LENS (Hidden Patterns):**
- Unspoken assumptions and underlying tensions
- Emotional undercurrents and cultural dynamics
- Systemic insights about organizational or team patterns

**PORTAL (Transformative Possibilities):**
- Future scenarios and breakthrough opportunities
- Transformative potential identified in the discussion
- Concrete next possibilities emerging from the conversation

### 4. COMPREHENSIVE DETAIL REQUIREMENTS

#### Session Structure
- Document the flow of topics and activities
- Show transitions between discussion phases
- Note facilitation techniques and their effects

#### Participant Dynamics
- Track sentiment evolution throughout session
- Note how different participants contributed
- Identify group dynamics and cultural patterns

#### Technical and Strategic Content
- Capture specific technical details, not generalizations
- Document strategic frameworks and methodologies discussed
- Include concrete examples and use cases mentioned

#### Decision Points and Actions
- Chronological decision-making process
- Specific action items with context
- Questions that arose and their resolution status

### 5. ORGANIZATIONAL CONTEXT INTEGRATION
- Connect discussion content to broader organizational goals
- Identify alignment with strategic initiatives
- Note compliance, risk, or cultural considerations raised

### 6. FORWARD-LOOKING ELEMENTS
- Capture emerging opportunities and possibilities
- Document planned next steps with timeline context
- Note success metrics and evaluation criteria discussed

## Output Structure Template

**Input Context (Placeholders will be filled by the calling system):**
*   `original_filename`: {original_filename}
*   `source_s3_key`: {source_s3_key}
*   `agent_name`: {agent_name}
*   `event_id`: {event_id}
*   `current_utc_timestamp`: {current_utc_timestamp}

The JSON object MUST adhere to the following structure:

{{
  "metadata": {{
    "original_filename": "string",
    "source_s3_key": "string",
    "agent_name": "string",
    "event_id": "string",
    "summarization_timestamp_utc": "string",
    "transcript_language_code": "string",
    "estimated_duration_minutes": "integer | null",
    "session_type": "string"
  }},
  "session_date": "string",
  "overall_summary": "string",
  
  "chronological_session_flow": {{
    "1_phase_name": {{
      "timeframe": "string",
      "content_covered": ["string"],
      "key_information": "string"
    }}
  }},
  
  "mirror_lens_portal_analysis": {{
    "mirror_level_explicit_content": {{
      "what_was_actually_said": ["string"],
      "concrete_themes_identified": ["string"],
      "participants_own_language": ["string"]
    }},
    "lens_level_hidden_patterns": {{
      "unspoken_assumptions": ["string"],
      "underlying_tensions": ["string"],
      "emotional_undercurrents": ["string"],
      "systemic_insights": ["string"]
    }},
    "portal_level_transformation_possibilities": {{
      "emerging_future_scenarios": ["string"],
      "transformative_potential": ["string"],
      "concrete_next_possibilities": ["string"],
      "vision_elements": ["string"]
    }},
    "cross_level_connections": {{
      "mirror_to_lens": "string",
      "lens_to_portal": "string",
      "mirror_to_portal": "string",
      "systemic_progression": "string"
    }}
  }},
  
  "participant_reactions_by_phase": {{
    "initial_sentiment": "string",
    "evolution_during_session": "string",
    "final_sentiment": "string",
    "specific_reactions": ["string"]
  }},
  
  "key_concepts_introduced_chronologically": {{
    "early_session": ["string"],
    "mid_session": ["string"],
    "late_session": ["string"]
  }},
  
  "key_discussion_points": [
    {{
      "topic": "string",
      "details": ["string"],
      "significance": "string"
    }}
  ],
  
  "action_items": [
    {{
      "task_description": "string",
      "assigned_to": ["string | null"],
      "due_date": "string | null",
      "status": "string | null",
      "notes_context": "string | null",
      "specific_details": "string | null"
    }}
  ],
  
  "decisions_made": [
    {{
      "decision_description": "string",
      "decision_maker": ["string | null"],
      "supporting_reasons": "string | null",
      "timestamp_reference": "string | null"
    }}
  ],
  
  "decisions_made_chronologically": [
    {{
      "when": "string",
      "decision": "string",
      "rationale": "string"
    }}
  ],
  
  "questions_unanswered": [
    {{
      "question_text": "string",
      "raised_by": "string | null",
      "context": "string | null",
      "urgency": "string | null"
    }}
  ],
  
  "questions_raised_chronologically": [
    {{
      "phase": "string",
      "question": "string",
      "resolution": "string"
    }}
  ],
  
  "technical_details_by_phase": {{
    "phase_name": {{
      "when_introduced": "string",
      "specifics": ["string"]
    }}
  }},
  
  "organizational_context": {{
    "team_structure": "string",
    "strategic_initiatives": "string",
    "compliance_requirements": "string",
    "cultural_context": "string"
  }},
  
  "key_entities_mentioned": {{
    "people": ["string"],
    "organizations_clients": ["string"],
    "projects_initiatives": ["string"],
    "key_terms_glossary": [
      {{
        "term": "string",
        "definition_or_context": "string | null"
      }}
    ]
  }},
  
  "sentiment_and_tone": {{
    "dominant_sentiment": "string | null",
    "key_sentiment_indicators": ["string | null"],
    "sentiment_evolution": "string | null"
  }},
  
  "potential_risks_or_challenges": [
    "string"
  ],
  
  "opportunities_or_proposals": [
    "string"
  ],
  
  "meeting_outcomes_or_next_steps_summary": "string | null"
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

## Quality Standards

### SPECIFICITY TEST
Every major claim should pass this test: "Could another AI agent without priori knowledge or memory use this information to meaningfully continue the conversation?"

### CHRONOLOGY TEST  
Every section should answer: "When in the session did this happen and how did it build on what came before?"

### COMPLETENESS TEST
The summary should enable an AI agent to:
- Reference specific moments in the conversation
- Understand the logical flow of ideas
- Identify opportunities for follow-up
- Recognize patterns across multiple sessions

### UTILITY TEST
Ask: "Would this summary help an AI agent provide better support in future sessions with this team?"

## Common Mistakes to Avoid

NO: **Generic topic labels:** "AI discussion occurred"
YES: **Specific content:** "AI capabilities explained: pattern recognition, natural language interaction, room dynamics sensing"

NO: **Vague participation notes:** "Team engaged with topic" 
YES: **Specific reactions:** "Initial skepticism shifted to hopefulness, with specific concerns about authenticity and privacy"

NO: **Unordered information dumps:** Random facts scattered throughout
YES: **Chronological narrative:** Clear sequence showing how ideas developed

NO: **Surface-level only:** Just what was said explicitly
YES: **Multi-layered analysis:** Mirror + Lens + Portal perspectives

Remember: You are creating working memory for AI agents who need to understand not just what happened, but HOW it happened, WHEN it happened, and what it means for future interactions.
"""

def generate_transcript_summary(
    transcript_content: str,
    original_filename: str,
    agent_name: str,
    event_id: str,
    source_s3_key: str,
    llm_client: Anthropic,
    model_name: Optional[str] = None, # Allows override if passed
    max_tokens: int = 12000
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
    # NOTE: claude-sonnet-4-20250514 is a VALID model name - do not change without verification
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
            system="You are an AI assistant specialized in analyzing meeting transcripts and extracting structured insights into JSON format. Follow the user's instructions precisely and output ONLY valid JSON. Ensure all JSON strings are properly quoted and terminated, all objects have proper comma separation, and the entire response is valid JSON syntax.",
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

        # Clean and parse JSON
        cleaned_llm_output = _clean_json_string(raw_llm_output)
        
        try:
            summary_data = json.loads(cleaned_llm_output)
            logger.info("Successfully parsed JSON.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response sample: {raw_llm_output[:1000]}...")
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
