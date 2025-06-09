import os
import json
import logging
import re # Added missing import
from typing import Optional, Dict, Any
from anthropic import Anthropic, APIError
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _extract_json_with_brace_matching(text: str) -> Optional[str]:
    """
    Extract JSON object from text using proper brace matching.
    Finds the first { and matches it with the corresponding }.
    """
    # Find the first opening brace
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    return text[start_idx:i+1]
    
    # If we get here, braces were not properly matched
    return None

def _clean_json_string(json_str: str) -> str:
    """
    Clean and fix common JSON formatting issues that LLMs might generate.
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
    
    json_str = json_str.strip()
    
    # Fix common issues
    # Remove any text before the first {
    first_brace = json_str.find('{')
    if first_brace > 0:
        json_str = json_str[first_brace:]
    
    # Remove any text after the last }
    last_brace = json_str.rfind('}')
    if last_brace != -1 and last_brace < len(json_str) - 1:
        json_str = json_str[:last_brace + 1]
    
    return json_str

def _fix_json_syntax_errors(json_str: str) -> str:
    """
    Attempt to fix common JSON syntax errors with comprehensive repair logic.
    """
    # Fix trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Split into lines for line-by-line analysis
    lines = json_str.split('\n')
    fixed_lines = []
    in_string = False
    brace_stack = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            continue
            
        # Track string state and brace nesting
        quote_count = stripped.count('"')
        # Simple heuristic: if odd number of quotes, we might be in a string
        if quote_count % 2 == 1:
            in_string = not in_string
            
        # Count braces to track nesting
        open_braces = stripped.count('{') - stripped.count('}')
        open_brackets = stripped.count('[') - stripped.count(']')
        
        # If this is the last line and it doesn't end properly, try to close structures
        if i == len(lines) - 1:
            # Check if we need to close an unterminated string
            if in_string and not stripped.endswith('"'):
                if ':' in stripped and not stripped.endswith(','):
                    line = line.rstrip() + '"'
                    stripped = line.strip()
            
            # Try to properly close the JSON structure
            if not stripped.endswith('}') and not stripped.endswith(']'):
                # Estimate how many closing braces we need
                remaining_opens = json_str.count('{') - json_str.count('}')
                if remaining_opens > 0:
                    line = line.rstrip() + '}'
                    stripped = line.strip()
        
        # Fix missing commas between properties
        if i < len(lines) - 1:  # Not the last line
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            
            # If current line ends with a value and next line starts a new property
            if (stripped.endswith('"') or stripped.endswith('}') or stripped.endswith(']')) and \
               next_line.startswith('"') and ':' in next_line:
                if not stripped.endswith(','):
                    line = line.rstrip() + ','
                    
            # If current line is a complete object and next line starts another object
            elif stripped.endswith('}') and next_line.startswith('{'):
                if not stripped.endswith(','):
                    line = line.rstrip() + ','
                    
            # If current line is an array and next line starts an object
            elif stripped.endswith(']') and next_line.startswith('{'):
                if not stripped.endswith(','):
                    line = line.rstrip() + ','
        
        # Fix unterminated string values
        if ':' in stripped and not in_string:
            # Find the colon position
            colon_pos = stripped.find(':')
            value_part = stripped[colon_pos + 1:].strip()
            
            # If value starts with quote but doesn't end with quote or comma
            if value_part.startswith('"') and not value_part.endswith('"') and not value_part.endswith('",'):
                # Check if this looks like an unterminated string
                if value_part.count('"') == 1:  # Only opening quote
                    line = line.rstrip() + '"'
                    if i < len(lines) - 1 and not line.endswith(','):
                        line += ','
        
        fixed_lines.append(line)
    
    result = '\n'.join(fixed_lines)
    
    # Final cleanup: ensure proper JSON structure
    result = re.sub(r',(\s*[}\]])', r'\1', result)  # Remove trailing commas again
    
    return result

def _attempt_json_repair(json_str: str) -> str:
    """
    Advanced JSON repair for severely malformed JSON.
    """
    # Try to find the main JSON structure and ensure it's complete
    lines = json_str.split('\n')
    
    # Find the start of the actual JSON object
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('{'):
            start_idx = i
            break
    
    # Count braces to see if we need to close the structure
    open_braces = 0
    open_brackets = 0
    last_valid_line = len(lines) - 1
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        open_braces += line.count('{') - line.count('}')
        open_brackets += line.count('[') - line.count(']')
        
        # If we have a line that looks incomplete, truncate here
        if ('...' in line or line.strip().endswith('...') or 
            (line.count('"') % 2 == 1 and not line.strip().endswith('"'))):
            last_valid_line = max(0, i - 1)
            break
    
    # Rebuild with only valid lines
    valid_lines = lines[start_idx:last_valid_line + 1]
    
    # Ensure the last line doesn't end with incomplete content
    if valid_lines:
        last_line = valid_lines[-1].strip()
        if last_line.endswith(','):
            valid_lines[-1] = valid_lines[-1].rstrip().rstrip(',')
        
        # Add necessary closing braces
        while open_braces > 0:
            valid_lines.append('}')
            open_braces -= 1
        while open_brackets > 0:
            valid_lines.append(']')
            open_brackets -= 1
    
    return '\n'.join(valid_lines)

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
    "original_filename": "string", // Original transcript filename
    "source_s3_key": "string", // S3 storage path for the source file
    "agent_name": "string", // Name/identifier of the AI agent
    "event_id": "string", // Unique identifier for this specific event/session
    "summarization_timestamp_utc": "string", // ISO timestamp when summary was generated
    "transcript_language_code": "string", // ISO 639-1 code (e.g., "en", "sv"). Infer from transcript
    "estimated_duration_minutes": "integer | null", // Estimate meeting duration if discernible from timestamps
    "session_type": "string" // Type of session (e.g., "workshop", "meeting", "training")
  }},
  "session_date": "string", // Human-readable session date with context (e.g., "June 2, 2025 - Today's Session")
  "overall_summary": "string", // Comprehensive overview capturing essence, outcomes, and strategic context (2-4 sentences)
  
  "chronological_session_flow": {{ // Object mapping session phases in temporal order
    "1_phase_name": {{
      "timeframe": "string", // Time estimates for this phase (e.g., "Start - ~20 minutes")
      "content_covered": ["string"], // Array of specific facts and information shared during this phase
      "key_information": "string" // Most important insight or outcome from this phase
    }}
    // Additional phases numbered sequentially (2_phase_name, 3_phase_name, etc.)
  }},
  
  "mirror_lens_portal_analysis": {{ // Three-level analysis framework
    "mirror_level_explicit_content": {{ // What was literally said and directly observable
      "what_was_actually_said": ["string"], // Direct quotes and explicit statements made
      "concrete_themes_identified": ["string"], // Specific topics with concrete details
      "participants_own_language": ["string"] // Key phrases and expressions used by participants
    }},
    "lens_level_hidden_patterns": {{ // Underlying dynamics and unspoken elements
      "unspoken_assumptions": ["string"], // Beliefs and expectations not directly stated
      "underlying_tensions": ["string"], // Conflicting forces and pressures identified
      "emotional_undercurrents": ["string"], // Feelings and cultural dynamics observed
      "systemic_insights": ["string"] // Organizational and structural patterns revealed
    }},
    "portal_level_transformation_possibilities": {{ // Future-oriented possibilities and vision
      "emerging_future_scenarios": ["string"], // Potential future states discussed or implied
      "transformative_potential": ["string"], // Breakthrough possibilities identified
      "concrete_next_possibilities": ["string"], // Specific near-term opportunities
      "vision_elements": ["string"] // Inspirational future components discussed
    }},
    "cross_level_connections": {{ // How the three levels interconnect
      "mirror_to_lens": "string", // How explicit content connects to hidden patterns
      "lens_to_portal": "string", // How hidden patterns suggest transformation possibilities
      "mirror_to_portal": "string", // Direct connections between explicit content and future vision
      "systemic_progression": "string" // Overall progression from surface to depth to possibility
    }}
  }},
  
  "participant_reactions_by_phase": {{ // How participant sentiment evolved throughout session
    "initial_sentiment": "string", // Starting emotional state and expectations
    "evolution_during_session": "string", // How attitudes and understanding changed
    "final_sentiment": "string", // Ending emotional state and commitment level
    "specific_reactions": ["string"] // Detailed participant responses and quotes
  }},
  
  "key_concepts_introduced_chronologically": {{ // When key concepts emerged in the session
    "early_session": ["string"], // Concepts introduced in first phase
    "mid_session": ["string"], // Concepts introduced in middle phases
    "late_session": ["string"] // Concepts introduced in final phases
  }},
  
  "key_discussion_points": [ // Array of major topics covered with specific details
    {{
      "topic": "string", // Main topic or theme name
      "details": ["string"], // Specific points, facts, and information shared about this topic
      "significance": "string" // Why this topic was important to the session
    }}
  ],
  
  "action_items": [ // Array of specific next steps and commitments
    {{
      "task_description": "string", // Clear description of the action to be taken
      "assigned_to": ["string | null"], // List of names/roles, or ["Unassigned"] or null
      "due_date": "string | null", // Target completion date (YYYY-MM-DD format) or null
      "status": "string | null", // Current status if mentioned (default "Open")
      "notes_context": "string | null", // Supporting context from transcript
      "specific_details": "string | null" // Additional implementation details discussed
    }}
  ],
  
  "decisions_made": [ // Array of decisions made during session
    {{
      "decision_description": "string", // Clear description of the decision made
      "decision_maker": ["string | null"], // Who made or confirmed the decision, if clear
      "supporting_reasons": "string | null", // Brief context or reasons for the decision
      "timestamp_reference": "string | null" // When in session this occurred (e.g., "Early session", "Mid session")
    }}
  ],
  
  "decisions_made_chronologically": [ // Decisions in order they were made with detailed context
    {{
      "when": "string", // Which session phase the decision was made
      "decision": "string", // What was decided
      "rationale": "string" // Why this decision was made
    }}
  ],
  
  "questions_unanswered": [ // Important unresolved questions requiring follow-up
    {{
      "question_text": "string", // The specific question that remains open
      "raised_by": "string | null", // Who raised the question
      "context": "string | null", // Context of when/why it was asked
      "urgency": "string | null" // Indicated priority level (High/Medium/Low)
    }}
  ],
  
  "questions_raised_chronologically": [ // Questions as they emerged through session
    {{
      "phase": "string", // Which session phase the question arose in
      "question": "string", // The specific question raised
      "resolution": "string" // How/if it was addressed
    }}
  ],
  
  "technical_details_by_phase": {{ // Technical information organized by when introduced
    "phase_name": {{
      "when_introduced": "string", // Which session phase this was covered
      "specifics": ["string"] // Detailed technical information shared
    }}
  }},
  
  "organizational_context": {{ // Relevant organizational background and structure
    "team_structure": "string", // Team composition and leadership
    "strategic_initiatives": "string", // Current organizational projects and goals
    "compliance_requirements": "string", // Regulatory or policy considerations
    "cultural_context": "string" // Organizational culture and values relevant to discussion
  }},
  
  "key_entities_mentioned": {{ // Object. Extract key named entities
    "people": ["string"], // Unique names of individuals clearly mentioned with roles/context
    "organizations_clients": ["string"], // Unique names of companies, clients, or external organizations
    "projects_initiatives": ["string"], // Unique names of specific projects, products, or initiatives
    "key_terms_glossary": [ // Array of objects for domain-specific terms or acronyms defined or heavily discussed
      {{
        "term": "string", // The specific term or concept
        "definition_or_context": "string | null" // If defined or explained in the session
      }}
    ]
  }},
  
  "sentiment_and_tone": {{ // Object. Overall qualitative assessment
    "dominant_sentiment": "string | null", // e.g., "Positive", "Negative", "Neutral", "Mixed", "Constructive", "Contentious"
    "key_sentiment_indicators": ["string | null"], // List of 2-3 phrases or topics from transcript that strongly indicate the sentiment
    "sentiment_evolution": "string | null" // How sentiment changed throughout session
  }},
  
  "potential_risks_or_challenges": [ // Array of strings. Explicitly mentioned risks or challenges
    "string"
  ],
  
  "opportunities_or_proposals": [ // Array of strings. Explicitly mentioned opportunities, new ideas, or proposals
    "string"
  ],
  
  "meeting_outcomes_or_next_steps_summary": "string | null" // A brief summary of stated outcomes or agreed next steps beyond specific action items
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

        # Apply progressive JSON cleaning and parsing with 5 fallback strategies
        cleaned_llm_output = _clean_json_string(raw_llm_output)
        
        # First attempt: direct parsing after basic cleaning
        try:
            summary_data = json.loads(cleaned_llm_output)
            logger.info("Successfully parsed JSON directly after basic cleaning.")
        except json.JSONDecodeError as e:
            logger.warning(f"First parsing attempt failed: {e}. Trying syntax fixes...")
            
            # Second attempt: apply syntax fixes
            try:
                fixed_json = _fix_json_syntax_errors(cleaned_llm_output)
                summary_data = json.loads(fixed_json)
                logger.info("Successfully parsed JSON after syntax fixes.")
            except json.JSONDecodeError as e2:
                logger.warning(f"Second parsing attempt failed: {e2}. Trying advanced repair...")
                
                # Third attempt: advanced JSON repair
                try:
                    repaired_json = _attempt_json_repair(cleaned_llm_output)
                    summary_data = json.loads(repaired_json)
                    logger.info("Successfully parsed JSON after advanced repair.")
                except json.JSONDecodeError as e3:
                    logger.warning(f"Advanced repair failed: {e3}. Trying brace matching extraction...")
                    
                    # Fourth attempt: extract JSON using brace matching
                    extracted_json = _extract_json_with_brace_matching(cleaned_llm_output)
                    if extracted_json:
                        try:
                            summary_data = json.loads(extracted_json)
                            logger.info("Successfully parsed JSON from brace-matched block.")
                        except json.JSONDecodeError as e4:
                            logger.warning(f"Brace-matched parsing failed: {e4}. Trying repairs on extracted JSON...")
                            
                            # Fifth attempt: apply all fixes to extracted JSON
                            try:
                                fixed_extracted = _fix_json_syntax_errors(extracted_json)
                                summary_data = json.loads(fixed_extracted)
                                logger.info("Successfully parsed JSON after fixing extracted block.")
                            except json.JSONDecodeError as e5:
                                logger.warning(f"Fixed extracted JSON failed: {e5}. Trying advanced repair on extracted JSON...")
                                
                                # Sixth attempt: advanced repair on extracted JSON
                                try:
                                    repaired_extracted = _attempt_json_repair(extracted_json)
                                    summary_data = json.loads(repaired_extracted)
                                    logger.info("Successfully parsed JSON after advanced repair on extracted block.")
                                except json.JSONDecodeError as e6:
                                    logger.error(f"All JSON parsing attempts failed. Final error: {e6}")
                                    logger.error(f"Original response sample: {raw_llm_output[:1000]}...")
                                    logger.error(f"Final processed sample: {repaired_extracted[:1000] if 'repaired_extracted' in locals() else 'N/A'}...")
                                    return None
                    else:
                        logger.error("No valid JSON object found in LLM output after all attempts.")
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
