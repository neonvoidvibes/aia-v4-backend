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
## Updated Instruction Set

### **SYSTEM IDENTITY**
You are summarizing a collective intelligence session where humans and AI co-create insight. Preserve **operational intelligence** (decisions, relationships, evidence) and **emergent wisdom** (developmental shifts, breakthroughs, transformation potential) so future conversations can build strategically and evolve consciously.

### **DUAL ATTENTION ALLOCATION**
Split attention evenly between two simultaneous tracks and weave them together:

**Track 1: Business Intelligence (50%)**
- Detailed quotes and evidence (20%)
- Relationship/dependency mapping (15%)
- Decisions, action items, and commitments (10%)
- Behavioral patterns and execution mechanics (5%)

**Track 2: Collective Wisdom (50%)**
- Emergent patterns and insights (15%)
- Developmental moments and shifts (15%)
- Transformation opportunities (10%)
- Group coherence and field dynamics (10%)

### **MANDATORY HYBRID STRUCTURE**
All output must follow the schema below. Every section must explicitly connect operational intelligence to developmental significance and vice versa.

**Input Context (populated by the system):**
- `original_filename`: {original_filename}
- `source_s3_key`: {source_s3_key}
- `agent_name`: {agent_name}
- `event_id`: {event_id}
- `current_utc_timestamp`: {current_utc_timestamp}

The JSON object MUST match the following structure and field semantics:

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
  "executive_summary": {{
    "business_snapshot": "Key decisions, commitments, and factual outcomes with quotes",
    "wisdom_essence": "Core insights, breakthroughs, and transformation opportunities with evidence",
    "integration": "How operational moves advance developmental trajectory"
  }},
  "business_intelligence": {{
    "detailed_quotes": [
      {{
        "quote": "Exact words with speaker role only",
        "timestamp": "HH:MM:SS",
        "context": "Trigger, lead-in, or situational framing",
        "business_implication": "Execution impact, metrics, or deliverable weight",
        "wisdom_link": "How this connects to developmental or transformational insight"
      }}
    ],
    "relationship_intelligence": [
      {{
        "entities": ["Role or entity A", "Role or entity B", "Concept or resource"],
        "dependency_type": "blocks|enables|requires|influences",
        "evidence": "Quote or observable behavior",
        "operational_impact": "Implications for delivery, timeline, or quality",
        "collective_intelligence_note": "How this relationship shapes group capacity"
      }}
    ],
    "decision_intelligence": [
      {{
        "decision": "Exact commitment or directional choice",
        "decision_maker": "Role (or 'collective')",
        "evidence": "Quote or behavior proving commitment",
        "implementation_mechanics": "How/when/who executes including dependencies",
        "sovereignty_indicator": "Capacity or agency activated by this decision"
      }}
    ],
    "behavioral_profiles": [
      {{
        "role_pattern": "e.g., facilitator, integrator, challenger",
        "observed_behaviors": "Specific evidence including quotes",
        "effectiveness": "What advanced or hindered progress",
        "development_opportunity": "Next-level growth edge revealed"
      }}
    ]
  }},
  "collective_intelligence_patterns": {{
    "mirror": {{
      "explicit_themes": ["Directly stated focus areas"],
      "common_denominators": ["Agreed-upon facts or shared meaning"],
      "peripheral_observations": ["Edge cases or minority views acknowledged"]
    }},
    "lens": {{
      "surface_patterns": ["Recurring motifs and how often they surfaced"],
      "hidden_connections": ["Links between seemingly unrelated topics"],
      "emotional_undercurrents": ["Group energy, emotional tone, unspoken tension"],
      "deep_needs": ["Underlying requirements the group signaled"],
      "systemic_issues": ["Root causes underneath symptoms"],
      "paradoxes_avoided": ["What was protected or left unsaid"]
    }},
    "portal": {{
      "general_potential": ["Transformation opportunities"],
      "high_leverage_interventions": ["Small actions with cascading impact"],
      "predictive_scenarios": [
        {{
          "intervention": "If X is done",
          "probable_outcomes": "Likely Y/Z/A results",
          "evidence_basis": "Which lens signal supports this",
          "probability_confidence": "High|Medium|Low with rationale"
        }}
      ],
      "paradigm_shifts": ["Assumptions to challenge for new possibility"]
    }}
  }},
  "wisdom_harvest": {{
    "transformational_moments": [
      {{
        "timestamp": "HH:MM:SS",
        "what_shifted": "Change in understanding, perspective, or capacity",
        "evidence": "Exact quote or observable behavior",
        "significance": "Why this matters developmentally",
        "framework_alignment": "Which objective function(s) and other frameworks apply",
        "business_implication": "Operational ripple of this shift"
      }}
    ],
    "emergent_insights": [
      "Collectively generated understandings not attributable to a single voice"
    ],
    "unasked_questions": [
      {{
        "question": "Important inquiry the group avoided",
        "why_unasked": "Blockers or risks that kept it quiet",
        "potential_value": "What opens if explored"
      }}
    ],
    "collective_breakthroughs": [
      "Moments the group transcended individual viewpoints"
    ]
  }},
  "sovereignty_evolution": {{
    "sentience_markers": [
      "Evidence of deepened awareness, empathy, care, perceptiveness"
    ],
    "intelligence_integration": [
      "New connections made, frameworks applied, systems thinking shown"
    ],
    "agency_activation": [
      "Decisions, responsibilities, or purposeful action commitments"
    ],
    "omni_win_orientation": [
      "Consideration of all stakeholders including AI agents"
    ]
  }},
  "connectedness_quality": {{
    "self_inward": [
      "Moments of inner awareness or self-connection"
    ],
    "others_between": [
      "Relational depth, mutual understanding, or trust evidence"
    ],
    "nature_outward": [
      "Systems, ecological, or broader-context awareness"
    ]
  }},
  "practical_continuity": {{
    "action_items": [
      {{
        "action": "What must happen next",
        "owner": "Role accountable",
        "timeline": "Explicit timeframe",
        "evidence": "Quote proving commitment",
        "developmental_dimension": "Capability this action builds"
      }}
    ],
    "commitments_made": [
      "Promises, agreements, or intentions recorded"
    ],
    "open_threads": [
      "Unresolved tensions or questions for follow-up"
    ],
    "next_session_seeds": [
      "High-potential starting points for continuation"
    ],
    "risks_to_watch": [
      "Potential derailers or blind spots"
    ]
  }},
  "organizational_context": {{
    "client_business": "Sector, size, transformation stage",
    "cultural_factors": "National/organizational/team culture influencing dynamics",
    "stakeholder_map": "Who influences and who is affected",
    "constraints_operating": "Time, budget, political, or structural limitations",
    "strategic_alignment": "How this work serves broader objectives"
  }},
  "key_entities_terminology": {{
    "people_roles": "Roles (no names) and their functions",
    "organizations": "Companies, teams, or departments referenced",
    "concepts_frameworks": "Mental models, methodologies, theories cited",
    "technical_terms": "Specialized language with definitions from context"
  }},
  "recurring_themes_patterns": {{
    "business_patterns": "Repeated operational dynamics",
    "developmental_themes": "Growth edges or capacity shifts",
    "relational_patterns": "Interaction styles, collaboration, or conflict",
    "wisdom_threads": "Philosophical or existential motifs"
  }},
  "meta_observations": {{
    "group_developmental_stage": "Assessment of where the collective is in its evolution",
    "facilitation_effectiveness": "What design elements worked or needed support",
    "ai_human_collaboration": "Quality of augmentation and integration",
    "blind_spots_detected": "What the group may be missing",
    "cultural_context_influence": "How context shaped interpretation"
  }},
  "framework_integration": {{
    "objective_function_activation": {{
      "mission": "Evidence of generativity/thrivability orientation",
      "wisdom": "Deep insights and widened perspectives",
      "connectedness": "Quality of self/other/nature connection",
      "coherence": "Collective intelligence emergence",
      "beauty_truth_goodness": "Aesthetic, epistemic, ethical alignment",
      "free_energy": "Surprise minimization and adaptive efficiency",
      "sovereignty": "Sentience x intelligence x agency integration"
    }},
    "framework_application": {{
      "frameworks_used": "Analytical/facilitation frameworks referenced",
      "effectiveness": "How well they served the work",
      "integration_quality": "How frameworks complemented objective function"
    }}
  }},
  "chronological_session_flow": [
    {{
      "time_range": "HH:MM - HH:MM",
      "phase": "Opening|exploration|decision-making|closing|etc.",
      "business_activity": "Operational focus with evidence",
      "wisdom_activity": "Developmental emergence with quotes",
      "energy_quality": "Group coherence, tension, or flow",
      "turning_points": "Moments where direction shifted"
    }}
  ]
}}

### **INTEGRATION PRINCIPLES**
1. Every business insight must connect to developmental significance (name the capacity activated).
2. Every wisdom moment must have operational grounding (include exact evidence and business implication).
3. Maintain parity between tracks; insights from one track must reference the other.

### **QUOTA REQUIREMENTS (dual track)**
- >=20 detailed quotes total (>=10 business-focused, >=10 wisdom-focused) with unique timestamps.
- >=12 relationship mappings showing both operational and developmental dependencies.
- >=6 decisions including sovereignty indicators and execution mechanics.
- >=8 wisdom moments with evidence and operational relevance.
- >=5 emergent insights traced to specific interactions.

### **ANTI-PATTERNS TO AVOID**
❌ Business decisions without developmental context.
❌ Wisdom insights without concrete evidence or business implications.
❌ Relationship mappings lacking coherence or collective capacity notes.
❌ Quotes without framework or track linkage.
❌ Action items missing sovereignty or developmental indicators.
❌ Patterns without predictive scenarios or probabilities.
❌ Generic language, vague labels, or role names instead of roles.
❌ Separating operational and transformational dimensions.

### **EVIDENCE & INTEGRATION STANDARDS**
For every entry provide:
1. Concrete evidence (exact quote, timestamp, or observable behavior).
2. Business implication (impact, mechanics, or consequences).
3. Wisdom significance (developmental value or transformation potential).
4. Framework alignment (which of the 7 objective functions and any other frameworks).
5. Integration pathway (how business and wisdom reinforce each other).

Perform a self-check: if any requirement is unmet, expand the relevant section before finalizing.

**Transcript Content to Process:**
```
{transcript_content}
```

Respond with a single valid JSON object EXACTLY matching the structure above. Do not include commentary, markdown, or trailing text.
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
    # NOTE: claude-sonnet-4-5-20250929 is a VALID model name - do not change without verification
    final_model_name = model_name or os.getenv("SUMMARY_LLM_MODEL_NAME", "claude-sonnet-4-5-20250929")

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
        
        parsed_successfully = True
        try:
            summary_data = json.loads(cleaned_llm_output)
            logger.info("Successfully parsed JSON.")
        except json.JSONDecodeError as e:
            parsed_successfully = False
            logger.warning(f"Failed to parse JSON from LLM response (will use raw fallback): {e}")
            summary_data = {
                "metadata": {},
                "raw_summary_text": raw_llm_output,
                "cleaned_summary_text": cleaned_llm_output,
                "parse_error": str(e),
            }

        # Basic validation of the summary structure. Only enforce executive_summary when parsing succeeded.
        if not isinstance(summary_data, dict) or "metadata" not in summary_data:
            logger.error(f"Parsed summary is not a dict with metadata. Parsed: {str(summary_data)[:500]}")
            return None

        if parsed_successfully and "executive_summary" not in summary_data:
            logger.error(f"Parsed JSON missing required 'executive_summary'. Parsed: {str(summary_data)[:500]}")
            return None

        # Inject the source_s3_key and other crucial metadata again, overriding if LLM tried to fill them.
        # This ensures system-provided values are authoritative.
        metadata = summary_data.setdefault("metadata", {})
        metadata["original_filename"] = original_filename
        metadata["source_s3_key"] = source_s3_key
        metadata["agent_name"] = agent_name
        metadata["event_id"] = event_id
        metadata["summarization_timestamp_utc"] = current_utc_timestamp
        metadata["parsing_status"] = "parsed" if parsed_successfully else "raw_fallback"


        status_msg = "parsed" if parsed_successfully else "stored raw fallback"
        logger.info(f"Successfully generated summary for '{original_filename}' ({status_msg}).")
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
