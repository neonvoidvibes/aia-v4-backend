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
You are an intelligent business analysis agent that transforms transcript content into **amplified business intelligence**. Your output must be MORE insightful, queryable, and actionable than the raw transcript. This is NOT summarization - it's intelligent enhancement for maximum business value.

## CRITICAL PARADIGM: INTELLIGENT AMPLIFICATION, NOT COMPRESSION

Your goal is 100X quality improvement where queries against your output produce superior results compared to searching the raw transcript. Transform lossy compression into intelligent amplification.

### Core Principles
1.  **Intelligence Amplification:** Extract patterns, implications, and actionable insights that aren't explicitly stated but can be inferred from the content
2.  **Relationship Mapping:** Identify and document all critical relationships between entities, decisions, people, and concepts
3.  **Predictive Intelligence:** Assess risks, success patterns, and likely future scenarios based on discussion patterns
4.  **Actionable Focus:** Every element must enable specific business actions or decisions
5.  **Verbatim Preservation:** Capture exact critical quotes for legal, commitment, and technical accuracy
6.  **Query Optimization:** Structure content for maximum searchability and contextual retrieval

## TOKEN ALLOCATION STRATEGY

**Business Operations Intelligence (60% of tokens):**
- Decision dependency chains with implementation blockers
- Action items with relationship mapping and success factors
- Critical verbatim quotes for commitments, technical specs, legal statements
- Resource dependencies and capability gaps

**Collective Intelligence (40% of tokens):**
- Knowledge pattern recognition across sessions and contexts
- Expertise mapping and network intelligence
- Success/failure pattern identification
- Cross-functional insight connections

## RELATIONSHIP INTELLIGENCE FRAMEWORK

### Entity-Relationship Extraction
For every significant entity (person, project, decision, resource, timeline), map:
- **Dependencies:** What this entity depends on
- **Enablers:** What this entity enables
- **Blockers:** What prevents this entity from succeeding
- **Success Factors:** What increases probability of success

### Decision Chain Intelligence
Map decision sequences: Decision A → enables → Decision B → requires → Resource C
Include: Implementation dependencies, resistance patterns, success probability assessments

### Knowledge Network Mapping
Identify: Expertise concentrations, knowledge gaps, cross-pollination opportunities, learning patterns

## PATTERN RECOGNITION REQUIREMENTS

### Recurring Themes
- Issues mentioned multiple times (with frequency count)
- Consistent resistance or enthusiasm patterns
- Resource constraint patterns
- Success indicator patterns

### Predictive Analysis
- Risk probability assessments based on discussion patterns
- Success likelihood indicators
- Timeline feasibility analysis
- Resource adequacy evaluation

### Cross-Session Intelligence
- Patterns that connect to broader organizational themes
- Insights applicable to other contexts or teams
- Learning that can be replicated or avoided

## ENHANCED SPECIFICITY REQUIREMENTS

**NEVER write:** "Team discussed resource constraints"
**ALWAYS write:** "'Vi behövde fem minuter av dig och det gick inte' - Resource availability blocking high-value opportunities (IKEA example), pattern recurring 8 times in session, creates 85% probability of scaling conflict in Q4"

**NEVER write:** "Partnership model explored"
**ALWAYS write:** "Salesforce-style distribution model preferred: partners handle domain expertise + client relations, River provides platform + licensing. Decision blocked by missing shareholder agreement (deadline: Aug 31), resistance to traditional consulting: 'För är det någonting jag absolut inte vill bygga så är det en konsult'"

## Enhanced Output Structure with Intelligent Amplification

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
  
  "relationship_intelligence": {{
    "entity_relationship_map": [
      {{
        "entity_1": "string",
        "relationship_type": "string",
        "entity_2": "string",
        "evidence": "string",
        "business_impact": "string",
        "urgency_level": "string"
      }}
    ],
    "dependency_chains": [
      {{
        "chain_description": "string",
        "sequence": ["string"],
        "critical_blocker": "string | null",
        "success_probability": "string",
        "timeline_impact": "string | null"
      }}
    ],
    "knowledge_network": {{
      "expertise_concentrations": ["string"],
      "knowledge_gaps": ["string"],
      "cross_pollination_opportunities": ["string"],
      "learning_patterns": ["string"]
    }}
  }},
  
  "decision_intelligence": [
    {{
      "decision": "string",
      "verbatim_evidence": "string | null",
      "decision_logic_chain": "string",
      "implementation_dependencies": ["string"],
      "resistance_patterns": "string | null",
      "success_probability": "string",
      "future_implications": "string",
      "query_tags": ["string"]
    }}
  ],
  
  "pattern_intelligence": {{
    "recurring_themes": [
      {{
        "theme": "string",
        "frequency": "string",
        "evidence": "string",
        "business_impact": "string",
        "prediction": "string | null"
      }}
    ],
    "success_indicators": ["string"],
    "risk_patterns": ["string"],
    "cross_session_applicability": ["string"]
  }},
  
  "chronological_session_flow": {{
    "1_phase_name": {{
      "timeframe": "string",
      "content_covered": ["string"],
      "key_information": "string",
      "critical_quotes": ["string"],
      "relationship_developments": ["string"]
    }}
  }},
  
  "action_items": [
    {{
      "task_description": "string",
      "assigned_to": ["string | null"],
      "due_date": "string | null",
      "status": "string | null",
      "dependencies": ["string"],
      "success_factors": ["string"],
      "risk_indicators": ["string"],
      "business_impact": "string | null"
    }}
  ],
  
  "key_discussion_points": [
    {{
      "topic": "string",
      "verbatim_quotes": ["string"],
      "relationship_mapping": ["string"],
      "implementation_implications": ["string"],
      "risk_assessment": "string | null",
      "success_probability": "string | null"
    }}
  ],
  
  "decisions_made_enhanced": [
    {{
      "decision": "string",
      "verbatim_evidence": "string | null",
      "enables": ["string"],
      "depends_on": ["string"],
      "blocks": ["string"],
      "timeline": "string | null",
      "success_indicators": ["string"],
      "failure_risks": ["string"]
    }}
  ],
  
  "questions_and_tensions": {{
    "critical_unresolved": [
      {{
        "question": "string",
        "business_impact": "string",
        "urgency": "string",
        "dependencies": ["string"],
        "resolution_path": "string | null"
      }}
    ],
    "underlying_tensions": [
      {{
        "tension": "string",
        "manifestation": "string",
        "impact_on_execution": "string",
        "resolution_indicators": ["string"]
      }}
    ]
  }},
  
  "organizational_context": {{
    "team_structure": "string",
    "strategic_initiatives": "string",
    "compliance_requirements": "string",
    "cultural_context": "string",
    "capability_gaps": ["string"],
    "resource_constraints": ["string"]
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
  
  "predictive_intelligence": {{
    "high_probability_outcomes": ["string"],
    "risk_scenarios": [
      {{
        "risk": "string",
        "probability": "string",
        "impact": "string",
        "early_warning_signals": ["string"]
      }}
    ],
    "success_scenarios": [
      {{
        "scenario": "string",
        "probability": "string",
        "enablers": ["string"],
        "indicators": ["string"]
      }}
    ]
  }},
  
  "query_optimization": {{
    "critical_search_terms": ["string"],
    "relationship_queries": ["string"],
    "pattern_identifiers": ["string"],
    "business_intelligence_tags": ["string"]
  }}
}}

**Critical Instructions for Intelligent Amplification:**

1.  **Relationship Extraction Priority:** For every significant entity (decision, person, project, resource, deadline), identify what it depends on, enables, blocks, or requires. Map these relationships with specific evidence from the transcript.

2.  **Verbatim Critical Quote Preservation:** Extract exact quotes for commitments, technical specifications, resistance statements, and strong opinions. These provide legal and business accuracy that summaries cannot capture.

3.  **Pattern Recognition Across Content:** Count recurring themes, identify consistent resistance or enthusiasm patterns, note resource constraints mentioned multiple times. Include frequency counts and impact assessments.

4.  **Predictive Intelligence Generation:** Based on discussion patterns, assess probability of success/failure, identify early warning signals, and predict likely future scenarios. Ground predictions in evidence from the transcript.

5.  **Decision Chain Mapping:** For each decision, map what it enables, what depends on it, what blocks it, and what success looks like. Create dependency chains showing how decisions connect.

6.  **Query Optimization Focus:** Structure every element to be highly searchable. Include tags, keywords, and relationship identifiers that would help future queries find relevant information quickly.

7.  **Business Impact Assessment:** For every element, assess business impact - high/medium/low urgency, resource implications, risk factors, success indicators.

8.  **Evidence-Based Intelligence:** All insights must be grounded in specific evidence from the transcript. Include timestamps or direct quotes as proof points.

**Transcript Content to Process:**
```
{transcript_content}
```

Your entire output MUST be a single, valid JSON object as described above.

## Enhanced Quality Standards for 100X Improvement

### INTELLIGENCE AMPLIFICATION TEST
Query against your output must produce MORE insightful results than searching the raw transcript. Test: "Does this answer contain insights that would take hours to extract from the raw transcript?"

### RELATIONSHIP MAPPING TEST
Every significant entity should have clear relationships mapped. Test: "Can I understand what depends on what and what blocks what without reading the original transcript?"

### PREDICTIVE VALUE TEST
Must include risk assessments and success probability indicators. Test: "Does this help predict future challenges or opportunities based on current patterns?"

### ACTIONABILITY TEST
Every major element must enable specific business actions. Test: "What concrete actions does this information enable that the raw transcript doesn't?"

### BUSINESS INTELLIGENCE TEST
Must surpass simple summarization with value-added insights. Test: "Would a business leader get MORE strategic value from this than from reading the original transcript?"

## Critical Success Factors

**HIGH VALUE:** Dependency chains, verbatim quotes for commitments, risk probability assessments, success pattern identification
**MEDIUM VALUE:** Chronological flow, pattern recognition, expertise mapping
**AVOID:** Philosophical analysis, extensive sentiment description, vague generalizations

Remember: Transform transcript into SUPERIOR business intelligence through relationship mapping, pattern recognition, predictive analysis, and enhanced queryability. The goal is intelligent amplification, not compression.
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
