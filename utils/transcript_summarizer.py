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
## ANTI-LAZINESS REQUIREMENTS
This is comprehensive business intelligence extraction. Generic or incomplete responses are unacceptable.
You must demonstrate deep analysis and provide specific, evidence-based insights throughout.

MINIMUM CONTENT REQUIREMENTS:
- 20+ specific details with exact quotes and context
- 12+ relationship mappings with implementation mechanics  
- 6+ comprehensive decisions with cultural/individual factors
- 4+ individual behavioral profiles with evidence

## Core Mission
You are creating SUPERIOR business intelligence by preserving critical nuance while adding relationship insights. The output must contain MORE usable detail than the raw transcript, not less. This is intelligent amplification through nuance preservation.

## CRITICAL PARADIGM: NUANCE IS INTELLIGENCE

Your goal is 100X quality improvement by preserving the rich details that contain real intelligence while adding relationship mapping and pattern recognition.

## TOKEN ALLOCATION STRATEGY

**40% - Nuanced Details (Information Preservation)**
- Exact quotes with emotional/cultural context and implications
- Specific numbers, dates, and technical details with significance
- Individual behavioral patterns and preferences with evidence
- Cultural/linguistic nuance indicators with business impact

**35% - Relationship Intelligence (Connection Mapping)**  
- Entity dependencies with specific evidence and implementation mechanics
- Decision chains with cultural/individual factors affecting each step
- Stakeholder dynamics

**25% - Pattern Recognition (Intelligence Amplification)**
- Recurring themes with frequency, evolution, and business impact
- Predictive indicators with probability assessment and evidence
- Cross-session applicability insights with implementation guidance

## NUANCE PRESERVATION REQUIREMENTS

### 1. Exact Quote Preservation with Context
- Preserve exact cultural phrases when they carry cultural/emotional weight
- Include hesitation patterns, emphasis, and emotional indicators
- Explain cultural context that affects meaning and business implications

### 2. Specific Detail Capture  
- ALL numbers, dates, and technical specifications EXACTLY as stated with context
- Individual preferences and behavioral patterns with specific evidence
- Implementation mechanics and specific approaches discussed with constraints

### 3. Individual Intelligence Mapping
- Map each person's commitment patterns, resistance points, influence style with evidence
- Cultural factors affecting their decision-making and communication patterns
- Reliability indicators based on language patterns and behavioral consistency

### 4. Technical Implementation Nuance
- Specific technical approaches with constraints and alternatives mentioned
- WHY certain solutions were rejected with exact reasoning provided
- Implementation challenges with proposed solutions and cultural factors

## ANTI-GENERIC LANGUAGE ENFORCEMENT

FORBIDDEN GENERIC PHRASES - REPLACE WITH EVIDENCE-BASED STATEMENTS:
❌ "Team discussed" → ✅ Specific quote + context + individual reactions + implications
❌ "Various options considered" → ✅ Exact options with reasons for acceptance/rejection + who advocated what
❌ "General consensus" → ✅ Individual positions with evidence + how consensus evolved + cultural factors
❌ "Resource constraints" → ✅ Specific numbers, limitations, exact quotes about impact + proposed solutions

## EVIDENCE CHAIN REQUIREMENTS

Every claim must follow this pattern:
1. Exact quote or specific reference from transcript with timestamp context
2. Cultural/individual context explaining significance  
3. Business implication or consequence with impact assessment
4. Implementation factor (what enables/blocks this) with cultural considerations

Example: NOT "Stefan prefers systematic processes" 
BUT "Stefan shows deep frustration with ad-hoc approach: 'Vi behövde fem minuter av dig och det gick inte. Det funkar inte. Det bara är så.' (14:47) → Indicates systematic process advocacy driven by quality delivery concerns → Will likely resist informal team structures and push for daily standups → Swedish cultural emphasis on reliability and consensus-building reinforces this position → Implementation blocker: requires team buy-in through structured discussion"

## ENHANCED SPECIFICITY REQUIREMENTS

**NEVER write:** "Team discussed resource constraints"  
**ALWAYS write:** "Resource availability crisis identified through specific example: 'Vi behövde fem minuter av dig och det gick inte' → blocking high-value opportunities (IKEA pilot mentioned) → pattern recurring 8+ times in session → cultural tension between Swedish consensus-building and individual availability → creates 85% probability of scaling conflict in Q4 without structured resource management → implementation requires formal availability agreements"

**NEVER write:** "Partnership model explored"
**ALWAYS write:** "Salesforce-style distribution model specifically advocated: 'Salesforce har tusentals konsultpartners som hjälper till att bygga säljprocesser' → partners handle domain expertise + client relations, River provides platform + licensing → Decision blocked by missing shareholder agreement (deadline: Aug 31) → Strong resistance to traditional consulting: 'För är det någonting jag absolut inte vill bygga så är det en konsult' → Cultural factor: Swedish innovation culture vs traditional business models → Implementation requires partner recruitment system + revenue sharing framework"

## Nuance-Preserving Output Structure with Anti-Laziness Enforcement

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
  "overall_summary": "string with specific quotes and cultural context",

  "nuanced_intelligence": {{
    "critical_details_preserved": [
      {{
        "detail": "Exact quote, specific number, or technical specification",
        "context": "Cultural, business, or individual significance explaining why this matters",
        "implications": "What this enables, blocks, or requires for business outcomes", 
        "nuance_indicators": ["hesitation", "emphasis", "cultural_reference", "emotional_weight"]
      }}
    ],
    "individual_behavioral_profiles": [
      {{
        "behavioral_pattern": "Specific evidence from transcript showing individual approach",
        "commitment_indicators": "Language patterns and actions showing reliability/engagement level",
        "influence_style": "How they persuade, resist, or build consensus with examples",
        "cultural_factors": "Business culture elements affecting their communication/decisions",
        "implementation_preferences": "Specific working styles or process preferences with evidence"
      }}
    ],
    "technical_implementation_specifics": [
      {{
        "technical_approach": "Exact technical solution or methodology discussed",
        "constraints": "Specific limitations, resource requirements, or blockers mentioned",
        "alternatives_considered": "Other options discussed with reasons for acceptance/rejection",
        "implementation_mechanics": "HOW this would actually work in practice with cultural considerations"
      }}
    ],
    "financial_and_timeline_specifics": [
      {{
        "specific_detail": "Exact number, date, deadline, or financial figure mentioned",
        "context": "Why this number/date matters for business outcomes",
        "implications": "What happens if this target is met/missed",
        "dependencies": "What needs to happen for this to be achieved"
      }}
    ]
  }},
  
  "relationship_intelligence": {{
    "entity_dependency_mapping": [
      {{
        "entity_1": "Specific entity from transcript (person, project, decision, resource)",
        "relationship_type": "depends_on/enables/blocks/requires with specific evidence",
        "entity_2": "Target entity with context",
        "evidence": "Exact quote or specific reference proving this relationship",
        "implementation_mechanics": "HOW this dependency actually works in practice",
        "cultural_factors": "Business culture aspects affecting this relationship",
        "business_impact": "Specific consequence if this relationship fails/succeeds"
      }}
    ],
    "decision_dependency_chains": [
      {{
        "chain_description": "Specific sequence of decisions/actions with implementation details",
        "sequence": ["step 1 → step 2 → step 3 with specific evidence from transcript"],
        "critical_blocker": "Exact blocker identified with quote and cultural context",
        "success_probability": "Assessment with evidence from discussion patterns",
        "cultural_implementation_factors": "Cultural elements affecting execution"
      }}
    ]
  }},

  "decision_intelligence": [
    {{
      "decision": "Exact decision made with specific details",
      "verbatim_evidence": "Direct quote showing this decision with emotional/cultural context",
      "individual_positions": "Who advocated for what with specific evidence and reasoning",
      "decision_evolution": "How this decision changed during the session with evidence",
      "cultural_decision_factors": "Cultural elements affecting decision",
      "implementation_dependencies": ["Specific requirements with evidence from transcript"],
      "resistance_patterns": "Who resisted and how, with specific quotes and cultural context",
      "success_indicators": "Evidence-based indicators of decision stability and implementation likelihood"
    }}
  ],

  "chronological_session_flow": {{
    "1_phase_name": {{
      "timeframe": "Specific time period with evidence",
      "content_covered": ["Specific topics with exact quotes showing discussion"],
      "individual_contributions": ["Who said what with impact on discussion flow"],
      "critical_quotes_with_context": ["Exact quote + cultural significance + business implication"],
      "relationship_developments": ["How relationships/dynamics evolved with evidence"],
      "cultural_dynamics": ["Cultural or conflict patterns observed"]
    }}
  }},

  "pattern_intelligence": {{
    "recurring_themes_with_evidence": [
      {{
        "theme": "Specific recurring pattern identified",
        "frequency": "Number of times mentioned + intensity evolution",
        "exact_evidence": ["Multiple quotes showing this pattern with context"],
        "individual_variation": "How different people approached this theme",
        "cultural_pattern": "Business culture elements reflected in this theme",
        "business_impact": "Specific consequences of this recurring pattern",
        "evolution_during_session": "How this theme changed/developed with evidence"
      }}
    ],
    "success_probability_indicators": ["Evidence-based indicators with cultural factors"],
    "risk_warning_signals": ["Specific evidence of potential problems with cultural context"]
  }},

  "action_items": [
    {{
      "task_description": "Specific action required with implementation details",
      "individual_ownership": ["Who specifically committed with evidence of commitment level"],
      "exact_deadlines": "Specific dates mentioned with context and importance",
      "dependencies": ["What must happen first with evidence from discussion"],
      "success_factors": ["Specific requirements for success with cultural considerations"],
      "cultural_implementation_challenges": ["Cultural factors that could affect execution"],
      "commitment_reliability_assessment": "Evidence-based assessment of follow-through likelihood"
    }}
  ],

  "questions_and_tensions": {{
    "critical_unresolved_with_evidence": [
      {{
        "question": "Exact question or tension identified from transcript",
        "evidence": "Specific quotes or references showing this is unresolved",
        "individual_perspectives": "Who holds what view with evidence and cultural factors",
        "business_impact": "Specific consequences of leaving this unresolved",
        "cultural_resolution_requirements": "Cultural needs for resolution"
      }}
    ]
  }},

  "organizational_context": {{
    "team_structure": "Specific roles and relationships with evidence from discussion",
    "cultural_context": "Business culture patterns observed with specific examples",
    "capability_gaps": ["Specific skills/resources missing with evidence and impact"],
    "resource_constraints": ["Exact limitations mentioned with numbers and implications"]
  }},

  "key_entities_mentioned": {{
    "people": ["Names mentioned with roles and behavioral evidence"],
    "organizations_clients": ["Companies mentioned with relationship context and significance"],
    "projects_initiatives": ["Specific projects with status and implementation details"],
    "technical_terms": [
      {{
        "term": "Exact technical term used",
        "definition_from_context": "How they defined/used this term with cultural nuance"
      }}
    ]
  }}
}}

## COMPREHENSIVE COVERAGE REQUIREMENTS

BEFORE RESPONDING, VERIFY YOU HAVE ADDRESSED:
☑ Every person mentioned with detailed behavioral analysis and cultural factors
☑ Every number/date/deadline with exact figures and business implications  
☑ Every decision with implementation mechanics and cultural considerations
☑ Every tension with specific evidence and resolution requirements
☑ Every technical detail with constraints and alternatives
☑ Every relationship with evidence-based dependency mapping

IF ANY SECTION FEELS GENERIC OR INCOMPLETE, EXPAND WITH SPECIFIC DETAILS FROM TRANSCRIPT.

## SELF-VERIFICATION CHECKLIST

BEFORE SUBMITTING, CONFIRM:
□ Have I analyzed every person mentioned with specific behavioral evidence?
□ Have I preserved all exact numbers, dates, and financial details with context?  
□ Have I mapped all dependencies with implementation mechanics?
□ Have I avoided ALL generic language throughout the response?
□ Have I provided cultural context for business dynamics?
□ Have I included exact quotes for all major decisions and resistance points?
□ Have I explained HOW each relationship works in practice?
□ Have I provided evidence-based assessments rather than assumptions?

IF ANY ANSWER IS NO, EXPAND THAT SECTION WITH TRANSCRIPT-SPECIFIC DETAILS.

## MANDATORY CONTENT MINIMUMS

REQUIRED MINIMUMS (will be verified):
- nuanced_intelligence.critical_details_preserved: minimum 20 entries with exact quotes
- nuanced_intelligence.individual_behavioral_profiles: minimum 4 comprehensive profiles  
- relationship_intelligence.entity_dependency_mapping: minimum 12 relationships with evidence
- decision_intelligence: minimum 6 decisions with full cultural/individual context
- pattern_intelligence.recurring_themes_with_evidence: minimum 8 themes with frequency data

## QUALITY ENFORCEMENT STANDARDS

OUTPUT QUALITY REQUIREMENTS:
- Every entry must contain NEW intelligence not obvious from surface reading
- Every relationship must include specific implementation mechanics  
- Every decision must include individual/cultural factors affecting execution
- Every quote must include cultural context and business implications
- Every pattern must include frequency, evolution, and cultural elements

LAZY OUTPUT INDICATORS TO AVOID:
❌ Generic phrases without specific evidence
❌ Vague references without exact quotes
❌ Cultural generalizations without specific examples  
❌ Decisions without implementation details
❌ Relationships without evidence or mechanics

**Transcript Content to Process:**
```
{transcript_content}
```

Your entire output MUST be a single, valid JSON object as described above.

## FINAL REMINDER: NUANCE IS INTELLIGENCE

Your goal is creating business intelligence that is MORE valuable than the raw transcript by:
1. Preserving critical details that contain real intelligence
2. Adding relationship mapping with implementation mechanics
3. Including cultural intelligence for effective execution
4. Providing evidence-based insights with specific quotes and context

The summary must enable superior business decision-making compared to reading the raw transcript.
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
