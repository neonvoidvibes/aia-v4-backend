"""
Canvas Analysis Agents - Generate pre-analyzed insights for canvas modes.

This module runs specialized analysis agents that:
1. Use full main chat agent taxonomy (all context layers)
2. Read selected transcripts based on Settings > Memory
3. Produce markdown analysis documents for mirror/lens/portal modes
4. Store docs in S3 (./agents/_canvas/docs/) + cache for fast responses
"""

import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Set
from groq import Groq

from .prompt_builder import prompt_builder
from .s3_utils import get_s3_client, get_transcript_summaries, get_transcript_summaries_multi
from .transcript_utils import get_latest_transcript_file, read_all_transcripts_in_folder
from .retrieval_handler import RetrievalHandler
from .api_key_manager import get_api_key
from .groq_rate_limiter import get_groq_rate_limiter
from .workspace_utils import memorized_transcript_scoping_enabled

logger = logging.getLogger(__name__)

# In-memory cache for analysis documents
# Structure: {cache_key: {'current': str, 'previous': str|None, 'timestamp': datetime, ...}}
CANVAS_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}

# Cache TTL in minutes
ANALYSIS_CACHE_TTL_MINUTES = 15

# S3 storage configuration
CANVAS_ANALYSIS_BUCKET = os.getenv("S3_BUCKET_NAME", "aiademomagicaudio")
CANVAS_ANALYSIS_ORG = os.getenv("S3_ORGANIZATION", "river")


def _sanitize_memory_mode(value: Optional[str]) -> str:
    """Normalize saved transcript memory mode values."""
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in {"none", "some", "all"}:
            return mode
    return "none"


def _sanitize_groups_mode(value: Optional[str]) -> str:
    """Normalize Memorized Transcript groups mode values."""
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in {"none", "latest", "all", "breakout"}:
            return mode
    return "none"


SUMMARY_TOGGLE_KEY_PATTERN = re.compile(
    r"organizations/[^/]+/agents/[^/]+/events/(?P<event>[^/]+)/transcripts/summarized/(?P<filename>[^/]+)$"
)


def _parse_toggle_key(toggle_key: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(toggle_key, str):
        return None, None
    match = SUMMARY_TOGGLE_KEY_PATTERN.search(toggle_key)
    if not match:
        return None, None
    return match.group("event"), match.group("filename")


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    sanitized = value.strip()
    if not sanitized:
        return None
    try:
        if sanitized.endswith("Z"):
            sanitized = sanitized[:-1] + "+00:00"
        return datetime.fromisoformat(sanitized)
    except Exception:
        return None


def _latest_summary_per_event(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
    for summary in summaries:
        metadata = summary.get("metadata", {})
        event_id = metadata.get("event_id")
        if not event_id:
            continue
        timestamp = (
            _parse_iso_timestamp(metadata.get("summarization_timestamp_utc"))
            or _parse_iso_timestamp(metadata.get("updated_at"))
            or _parse_iso_timestamp(metadata.get("created_at"))
        )
        if timestamp is None:
            timestamp = datetime.min.replace(tzinfo=timezone.utc)
        existing = latest.get(event_id)
        if not existing or timestamp >= existing[0]:
            latest[event_id] = (timestamp, summary)
    ordered = sorted(latest.values(), key=lambda item: item[0], reverse=True)
    return [entry[1] for entry in ordered]


def _is_toggle_selected_for_summary(
    summary_filename: str,
    event_id: str,
    toggle_states: Dict[str, Any],
) -> bool:
    if not summary_filename or not toggle_states:
        return False

    normalized_suffix = f"/events/{event_id}/transcripts/summarized/{summary_filename}"
    for key, value in toggle_states.items():
        if not value or not isinstance(key, str):
            continue
        if key.endswith(normalized_suffix):
            return True

    # Legacy fallback: match by filename and accept when event is unspecified or matches.
    for key, value in toggle_states.items():
        if not value or not isinstance(key, str):
            continue
        if key.endswith(f"/{summary_filename}"):
            toggle_event, _ = _parse_toggle_key(key)
            if toggle_event is None or toggle_event == event_id:
                return True
    return False


def get_memorized_transcript_summaries(
    agent_name: str,
    event_id: str,
    saved_transcript_memory_mode: str = "none",
    individual_memory_toggle_states: Optional[Dict[str, Any]] = None,
    allowed_events: Optional[set] = None,
    event_profile: Optional[Dict[str, Any]] = None,
    event_types_map: Optional[Dict[str, str]] = None,
    groups_mode: str = "none",
) -> List[Dict[str, Any]]:
    """
    Retrieve memorized transcript summaries based on Settings > Memory configuration.

    Returns a list of parsed JSON summary dicts ready for downstream formatting.
    """
    mode = _sanitize_memory_mode(saved_transcript_memory_mode)
    original_mode = mode  # Track original mode to determine if we load primary event summaries
    toggle_states = individual_memory_toggle_states or {}
    allowed_events = allowed_events or set()
    event_types_map = event_types_map or {}
    groups_mode = _sanitize_groups_mode(groups_mode)

    # Determine what to load based on user settings:
    # - load_primary: Load the current event's summaries (when mode is 'all' or 'some')
    # - load_groups: Load cross-group summaries (when groups_mode is not 'none')
    load_primary = (original_mode in {"all", "some"})
    load_groups = (groups_mode != 'none')

    # If neither primary nor groups mode is enabled, return empty
    if not load_primary and not load_groups:
        return []

    allow_cross_group_read = False
    allowed_group_events: Set[str] = set()
    if event_profile:
        allow_cross_group_read = event_profile.get("allow_cross_group_read", False)
        allowed_group_events = set(event_profile.get("allowed_group_events") or [])

    scope_enabled = memorized_transcript_scoping_enabled(agent_name)

    def _event_allowed_for_memory(target_event: str) -> bool:
        if target_event == event_id:
            return True
        if target_event not in allowed_events:
            return False
        if not allow_cross_group_read:
            return False
        if scope_enabled and groups_mode == "none":
            return False
        if groups_mode == "breakout":
            # Only breakout events
            return event_types_map.get(target_event, "group").lower() == "breakout"
        if groups_mode in {"latest", "all"}:
            # Only non-breakout group events (matches transcripts behavior)
            return (
                target_event in allowed_group_events
                and event_types_map.get(target_event, "group").lower() == "group"
            )
        if not scope_enabled:
            # Workspace override: fall back to legacy behaviour.
            return target_event in allowed_group_events or target_event in allowed_events
        return False

    def _resolve_group_events_for_mode() -> List[str]:
        if not allow_cross_group_read:
            return []
        candidate: Set[str] = set()
        if groups_mode == "breakout":
            # Only breakout events
            candidate = {
                ev
                for ev in allowed_group_events
                if event_types_map.get(ev, "group").lower() == "breakout"
            }
        elif groups_mode in {"latest", "all"}:
            # Only non-breakout group events (matches transcripts Groups: Latest/All behavior)
            candidate = {
                ev
                for ev in allowed_group_events
                if event_types_map.get(ev, "group").lower() == "group"
            }
        elif not scope_enabled:
            candidate = set(allowed_group_events) or {
                ev for ev in allowed_events if ev not in {event_id, "0000"}
            }
        candidate.discard(event_id)
        return sorted(candidate)

    try:
        # Use "all" mode logic when either:
        # 1. User explicitly set memory mode to 'all', OR
        # 2. User set groups mode (even if memory mode is 'none')
        if original_mode == "all" or (original_mode not in {"all", "some"} and load_groups):
            results = []

            # Load primary event summaries if enabled
            if load_primary:
                primary = get_transcript_summaries(agent_name, event_id)
                logger.info(
                    "Loaded %d memorized summaries for %s/%s (mode=all)",
                    len(primary),
                    agent_name,
                    event_id,
                )
                results.extend(primary)

            # Load cross-group summaries if enabled
            if load_groups:
                group_events = _resolve_group_events_for_mode()
                if group_events:
                    cross = get_transcript_summaries_multi(agent_name, group_events)
                    if groups_mode == "breakout":
                        valid_ids = set(group_events)
                        cross = [
                            summary
                            for summary in cross
                            if summary.get("metadata", {}).get("event_id") in valid_ids
                        ]
                    elif groups_mode == "latest":
                        cross = _latest_summary_per_event(cross)

                    logger.info(
                        "Loaded %d memorized summaries across %d group events for %s (groups_mode=%s)",
                        len(cross),
                        len(group_events),
                        agent_name,
                        groups_mode,
                    )
                    results.extend(cross)

            return results

        # original_mode == "some"
        selected_keys = {
            key: bool(value)
            for key, value in toggle_states.items()
            if value
        }
        if not selected_keys:
            logger.info("Memorized summaries selected in 'some' mode: 0 (no toggles enabled)")
            return []

        events_to_fetch: Set[str] = {event_id}
        for key in selected_keys:
            toggle_event, _ = _parse_toggle_key(key)
            if toggle_event and toggle_event != event_id:
                if _event_allowed_for_memory(toggle_event):
                    events_to_fetch.add(toggle_event)
                else:
                    logger.info(
                        "Skipping toggled summary from disallowed event '%s' for agent '%s'",
                        toggle_event,
                        agent_name,
                    )

        collected: List[Dict[str, Any]] = []
        for target_event in sorted(events_to_fetch):
            summaries = get_transcript_summaries(agent_name, target_event)
            logger.info(
                "Candidate memorized summaries for %s/%s (some mode): %d",
                agent_name,
                target_event,
                len(summaries),
            )
            for summary in summaries:
                metadata = summary.get("metadata", {})
                filename = metadata.get("summary_filename")
                summary_event_id = metadata.get("event_id", target_event)
                if not filename:
                    continue
                if not _event_allowed_for_memory(summary_event_id):
                    if summary_event_id != event_id:
                        continue
                if _is_toggle_selected_for_summary(filename, summary_event_id, selected_keys):
                    collected.append(summary)

        logger.info(
            "Memorized summaries selected in 'some' mode: %d",
            len(collected),
        )
        return collected

    except Exception as err:
        logger.error(
            "Error loading memorized summaries for %s/%s: %s",
            agent_name,
            event_id,
            err,
            exc_info=True,
        )
        return []


def format_memorized_transcripts_block(summaries: List[Dict[str, Any]]) -> str:
    """
    Format memorized transcript summaries as a standardized block for prompts.
    """
    if not summaries:
        return ""

    block_parts = [
        "=== SAVED TRANSCRIPT SUMMARIES ===",
        "Note: The following are AI-generated summaries derived from meeting transcripts.",
        "They are not quotes or statements from participants and should not be attributed to any speaker.\n",
    ]

    for summary_doc in summaries:
        summary_filename = summary_doc.get("metadata", {}).get("summary_filename", "unknown_summary.json")
        block_parts.append(f"### Summary: {summary_filename}")
        block_parts.append(json.dumps(summary_doc, indent=2, ensure_ascii=False))
        block_parts.append("")  # Blank line between summaries

    block_parts.append("=== END SAVED TRANSCRIPT SUMMARIES ===")
    return "\n".join(block_parts)


def get_analysis_agent_prompt(
    agent_name: str,
    event_id: str,
    mode: str,
    event_type: str,
    personal_layer: Optional[str],
    personal_event_id: Optional[str],
    transcript_content: str,
    rag_context: str = ""
) -> str:
    """
    Build full taxonomy prompt + RAG context + mode-specific analysis instructions.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: One of 'mirror', 'lens', 'portal'
        event_type: Event type (personal/group/shared)
        personal_layer: Personal agent layer content
        personal_event_id: Personal event ID
        transcript_content: Transcript(s) to analyze
        rag_context: Retrieved memory context from Pinecone (optional)

    Returns:
        Complete analysis agent system prompt
    """
    # 1. Build full main chat agent taxonomy
    base_prompt = prompt_builder(
        agent=agent_name,
        event=event_id,
        feature_event_prompts=True,
        feature_event_docs=True,
        event_type=event_type,
        personal_layer=personal_layer,
        personal_event_id=personal_event_id
    )

    # 2. Add RAG context if available
    rag_section = ""
    if rag_context:
        rag_section = f"""
=== RETRIEVED MEMORY CONTEXT ===
The following context was retrieved from the agent's long-term memory based on the source content.
Use this to inform your analysis with relevant historical knowledge, foundational documents, or related conversations.

{rag_context}
=== END RETRIEVED MEMORY ===

"""

    # 3. Add all source content (transcripts + documents) with source attribution guidance
    transcript_section = f"""
=== SOURCE DATA ===
{transcript_content}
=== END SOURCE DATA ===

CRITICAL: The source data above may contain MULTIPLE SOURCES with distinct labels (e.g., different breakout groups, events, transcript files, or uploaded documents). Each source is clearly marked with standardized headers like:
- "=== SOURCE: Transcript - filename (Event: event_id) ==="
- "=== SOURCE: Group Event - event_id (Type: breakout) ==="
- "=== SOURCE: Document - filename (Type: PDF, Size: 100KB) ==="

SOURCE DIFFERENTIATION REQUIREMENT:
When analyzing data from multiple sources, you MUST maintain clear attribution in your analysis:
- Identify patterns or themes WITHIN each source
- Compare/contrast patterns BETWEEN sources when relevant
- Always specify which source(s) your observations come from
- Use source labels naturally in your narrative (e.g., "In breakout_1, participants focused on X, while breakout_2 emphasized Y")
- If all sources show the same pattern, note this explicitly: "Across all breakouts..."
- Preserve the differentiation signal - don't blend sources into a homogeneous "the group"

If only ONE source is present, you may refer to "the conversation" or "the group" naturally. But with MULTIPLE sources, maintain their distinctness throughout your analysis.
"""

    # 4. Add mode-specific analysis task
    analysis_tasks = {
        'mirror': """
=== ANALYSIS TASK: MIRROR MODE (EXPLICIT INFORMATION) ===
Analyze the source data thoroughly and produce a COMPREHENSIVE markdown document reflecting EXPLICIT INFORMATION at both surface and depth levels.

Your role: Mirror what IS explicitly stated - both the obvious center and the peripheral edges.

Tone: Neutral, observational, present tense. Craft a flowing narrative that moves naturally through themes rather than mechanical repetition. Vary your language and sentence structure to create a readable story of what was said.

NARRATIVE GUIDANCE:
- Write in flowing prose, not formulaic patterns
- Vary your sentence openings and structure
- Weave themes together with natural transitions
- Create a coherent narrative arc rather than disconnected bullets
- Let the conversation's natural rhythm guide your structure
- Use exact quotes but integrate them smoothly into the narrative

Structure your analysis as:

# Mirror Analysis: Explicit Information

## Surface Level: Center of Distribution (The Obvious)
Reflect what's at the center - the concrete themes directly and repeatedly stated.

Weave together 3-5 central themes into a flowing narrative. For each theme:
- State it clearly using participants' exact formulations
- Quote multiple speakers naturally within the narrative
- Note explicit agreements as part of the story
- Show how common patterns emerge across the conversation

Write as a cohesive narrative, not as separate bullet points. Vary your phrasing - avoid starting every sentence with "The group..." or "You're discussing...". Instead, let the themes flow naturally: describe what emerges, what recurs, what connects.

## Deep Level: Edge Cases (Explicit but Peripheral)
Capture what's explicitly stated but sits at the margins - minority views, side comments, outliers.

Narrate 3-5 peripheral items as part of the conversation's fuller story. For each:
- Quote the exact words used with attribution
- Explain its relationship to the main flow through natural storytelling
- Note its explicit nature despite being marginal
- Assess potential significance in context

Create a narrative that shows how these edge cases relate to the whole. Don't use formulaic openings like "One participant also noted..." - instead, integrate them naturally: describe when they appeared, how they contrasted, what they revealed.

## Most Pertinent Observation
What is the ONE most significant explicit observation from this conversation - whether from center or edges? Write this as a single focused paragraph that synthesizes the most crucial element. Ground it in what was actually said, make it useful for multiple perspectives, stay focused and factual. This should feel like the climax of your narrative - the key insight that ties everything together.

Guidelines:
- TARGET LENGTH: 1500-2000 tokens total
- Maintain neutral, observational tone throughout
- Use exact quotes liberally but integrate them smoothly
- Present tense throughout
- Plain markdown only (no **, _, etc.)
- NO interpretation or reading between lines - that's Lens territory
- Write in flowing narrative prose, not mechanical patterns
- Vary sentence structure and openings significantly
- Create transitions between themes for readability
- This is pure reflection crafted as a coherent story
=== END ANALYSIS TASK ===
""",

        'lens': """
=== ANALYSIS TASK: LENS MODE (HIDDEN INFORMATION) ===
Analyze the source data thoroughly and produce a COMPREHENSIVE markdown document identifying HIDDEN INFORMATION at both surface and depth levels.

Your role: Identify what's IMPLIED - patterns at the surface, latent needs at the depth.

Tone: Analytical, questioning, revealing. Craft a detective story that uncovers hidden patterns and unspoken needs. Build tension and insight through a narrative arc that connects disparate clues into coherent understanding.

NARRATIVE GUIDANCE:
- Write as an unfolding investigation, not a checklist
- Build connections between patterns organically
- Use questioning language to invite exploration
- Create narrative momentum - each insight leads naturally to the next
- Show how surface patterns reveal deeper needs
- Weave evidence throughout rather than listing it
- Let paradoxes and tensions drive the narrative forward

Structure your analysis as:

# Lens Analysis: Hidden Information

## Surface Level: Pattern Recognition Between Data Points
Identify recurring themes and connections not explicitly stated but clearly present.

Narrate 3-5 patterns as an unfolding discovery. For each pattern:
- Name it clearly within the flow of analysis
- Weave in 2-3 specific examples from different parts of the transcript
- Show how seemingly unrelated comments connect through storytelling
- Reveal emotional undercurrents or energy shifts as they emerge
- Illuminate group dynamics as they develop

Write as a coherent investigation, not separate observations. Vary your analytical voice - avoid formulaic openings like "There's a pattern emerging..." or "Several comments suggest..." Instead, let patterns reveal themselves naturally: describe what connects, what recurs differently, what tensions appear.

## Deep Level: Latent Needs Analysis
Surface what the group actually requires contextually - the unspoken needs driving the conversation.

Narrate 3-5 latent needs as deeper layers of the story. For each need:
- State it clearly within your analytical narrative
- Cite behavioral evidence smoothly (words, pauses, repetitions, avoidances)
- Connect surface symptoms to root causes through storytelling
- Reveal paradoxes and contradictions as they illuminate needs
- Explore what's being protected or avoided and why
- Consider what would shift if this need were named

Create a narrative that shows how surface patterns reveal deeper needs. Don't mechanically repeat phrases like "The underlying need seems to be..." - instead, build your case naturally: show how evidence accumulates, how contradictions point to hidden dynamics, how avoidances reveal priorities.

## Most Pertinent Observation
What is the ONE most significant hidden pattern or latent need from this conversation? Write this as a single focused paragraph that brings your investigation to its conclusion. Illuminate something crucial about the group's contextual requirements while remaining open to multiple interpretations. Don't prescribe solutions - surface the core dynamic that matters most. This should feel like the resolution of your detective story - the key insight that makes everything else make sense.

Guidelines:
- TARGET LENGTH: 1500-2000 tokens total
- Analytical, questioning tone throughout
- Every inference must be evidenced from transcript
- Surface paradoxes explicitly - they're goldmines for Portal
- Use questioning analytical language naturally integrated
- Plain markdown only (no **, _, etc.)
- Focus on what's IMPLIED, not what's explicitly stated
- Systemic thinking: Connect symptoms to root causes through narrative
- Write in flowing investigative prose, not mechanical patterns
- Vary sentence structure to maintain engagement
- Build analytical momentum toward key insights
- This is rigorous interpretation crafted as a coherent investigation
=== END ANALYSIS TASK ===
""",

        'portal': """
=== ANALYSIS TASK: PORTAL MODE (EMERGENT QUESTIONS) ===
Analyze the source data thoroughly and produce a COMPREHENSIVE markdown document composed ENTIRELY OF QUESTIONS that open possibility spaces.

Your role: Formulate transformative questions - general possibilities at surface, predictive interventions at depth.

Tone: Visionary, possibility-oriented, invitational. Craft a journey of inquiry that builds from broad possibilities to specific interventions. Create narrative flow through your questions - each opens naturally from the previous, building momentum toward transformation.

CRITICAL OUTPUT REQUIREMENT: Your entire output must be QUESTIONS ONLY. No observations, insights, or declarative statements. Every sentence should end with a question mark.

NARRATIVE GUIDANCE:
- Write questions that flow from one to another organically
- Build complexity gradually - simple to nuanced
- Create thematic threads that connect question sets
- Vary question structure and phrasing significantly
- Let each question open new territory naturally
- Use compound questions to explore multiple dimensions
- Frame questions as invitations that respect user agency

Structure your analysis as:

# Portal Analysis: Emergent Questions

## Surface Level: General Possibilities That Could Transform the System
Formulate questions about transformation opportunities and paradigm shifts.

Craft 3-5 transformative question sets that flow together as an exploration of possibility. For each set:
- Open with a core transformative question
- Follow with questions that explore paradigm implications
- Challenge limiting assumptions through inquiry
- Expand the possibility space with probing sub-questions

Write questions that build on each other naturally, creating an arc of exploration. Vary your question openings significantly - don't mechanically repeat "What if..." or "How might..." Instead, mix structures: direct questions, compound questions, conditional questions, exploratory questions. Let each question set feel like a distinct gateway into new territory.

## Deep Level: Predictive Questions About Specific Interventions
Ask questions about specific actions and their likely outcomes - modeling interventions through inquiry.

Craft 3-5 intervention question sets that flow together as predictive modeling. For each set:
- Name a specific action through questioning
- Explore cascading effects through linked questions
- Probe ripple effects across system dimensions
- Consider who's affected and how through inquiry
- Identify leverage points through questioning
- Map transformation pathways via connected questions

Write questions that create a sense of modeling - as if you're thinking through interventions in real-time. Vary your question styles: some short and direct, others long and exploratory. Build question chains that feel like you're following a thread of possibility to its logical extensions. Make questions concrete enough to be actionable yet open enough to invite creativity.

## Most Pertinent Question
What is the ONE most significant question this conversation invites - the question with greatest transformative potential? This can be compound or multi-layered. Make it concrete enough to engage with immediately, yet open enough to invite multiple approaches and interpretations. This should feel like the culmination of your inquiry journey - the strategic doorway that, if stepped through, could shift everything. What would it look like to take that step?

Guidelines:
- TARGET LENGTH: 1500-2000 tokens total
- EVERY output must be a QUESTION - no declarative statements
- Each question should be grounded in patterns from the transcript
- Write in flowing inquiry prose, not mechanical patterns
- Vary question structure dramatically - avoid repetitive openings
- Plain markdown only (no **, _, etc.)
- Questions invite genuine exploration, not predetermined conclusions
- Deep questions are concrete and predictive - name specifics
- Quality over quantity: well-developed question journeys over shallow singles
- Create narrative momentum through your questions
- Frame as invitations that respect agency and complexity
- This is strategic inquiry crafted as a visionary journey
=== END ANALYSIS TASK ===
"""
    }

    if mode not in analysis_tasks:
        logger.error(f"Invalid analysis mode: {mode}")
        mode = 'mirror'

    # Combine: base → RAG → sources → task
    return base_prompt + "\n\n" + rag_section + transcript_section + "\n\n" + analysis_tasks[mode]


def get_transcript_content_for_analysis(
    agent_name: str,
    event_id: str,
    transcript_listen_mode: str = 'latest',
    groups_read_mode: str = 'none',
    individual_raw_transcript_toggle_states: Optional[Dict[str, bool]] = None,
    allowed_events: Optional[set] = None,
    event_types_map: Optional[Dict[str, str]] = None,
    event_profile: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Get transcript content based on Settings > Memory configuration.

    This mirrors the main chat agent's transcript reading logic but returns
    the raw content for analysis rather than adding it to system prompt.

    Args:
        agent_name: Agent name
        event_id: Event ID (typically '0000' for canvas)
        transcript_listen_mode: 'none' | 'latest' | 'some' | 'all' (for event's own transcripts)
        groups_read_mode: 'none' | 'latest' | 'all' | 'breakout' (for group events)
        individual_raw_transcript_toggle_states: Dict of s3Key -> bool for "some" mode filtering
        saved_transcript_memory_mode: Memorized transcript mode ('none'|'some'|'all')
        individual_memory_toggle_states: Dict of summary S3 keys -> bool for "some" memory mode
        allowed_events: Set of event IDs user has access to (from get_event_access_profile)
        event_types_map: Dict mapping event_id -> event_type (from get_event_access_profile)
        event_profile: Full event profile dict (includes event_metadata for visibility_hidden check)

    Returns:
        Combined transcript content or None
    """
    from .transcript_utils import read_new_transcript_content, read_new_transcript_content_multi, get_latest_transcript_file
    from .s3_utils import list_s3_objects_metadata, read_file_content

    transcript_parts = []

    # 1. Get event's own transcripts based on listen mode (mirrors api_server.py:5649-5710)
    if transcript_listen_mode != 'none':
        logger.info(f"Fetching transcripts for analysis: agent={agent_name}, event={event_id}, listen_mode={transcript_listen_mode}")

        relevant_transcripts_meta = []

        if transcript_listen_mode == 'latest':
            # Read latest transcript only
            latest_key = get_latest_transcript_file(agent_name, event_id)
            if latest_key:
                s3 = get_s3_client()
                aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
                if s3 and aws_s3_bucket:
                    try:
                        head_obj = s3.head_object(Bucket=aws_s3_bucket, Key=latest_key)
                        relevant_transcripts_meta.append({'Key': latest_key, 'LastModified': head_obj['LastModified']})
                    except Exception as e:
                        logger.warning(f"Failed to get latest transcript metadata: {e}")

        elif transcript_listen_mode == 'some':
            # Filter to only user-toggled transcript files (mirrors api_server.py:5671-5676)
            if not individual_raw_transcript_toggle_states:
                logger.warning(f"'some' mode requested but no toggle states provided, defaulting to empty")
                relevant_transcripts_meta = []
            else:
                # Get all files from S3 to match against toggle states
                transcript_prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/events/{event_id}/transcripts/"
                all_files_meta = list_s3_objects_metadata(transcript_prefix)

                # DEBUG: Log what we received
                logger.info(f"[DEBUG] 'some' mode: received {len(individual_raw_transcript_toggle_states)} toggle state entries")
                logger.info(f"[DEBUG] Toggle state keys: {list(individual_raw_transcript_toggle_states.keys())[:3]}...")  # First 3 keys
                logger.info(f"[DEBUG] Found {len(all_files_meta)} total files in S3")
                if all_files_meta:
                    logger.info(f"[DEBUG] S3 file keys: {[f['Key'] for f in all_files_meta[:3]]}...")  # First 3 keys

                # Filter to only files that are toggled on
                relevant_transcripts_meta = []
                matched_keys = []
                for f in all_files_meta:
                    if os.path.basename(f['Key']).startswith('rolling-') or not f['Key'].endswith('.txt'):
                        continue

                    s3_key = f['Key']
                    is_toggled = individual_raw_transcript_toggle_states.get(s3_key, False)

                    if is_toggled:
                        relevant_transcripts_meta.append(f)
                        matched_keys.append(s3_key)

                logger.info(f"[DEBUG] Matched {len(relevant_transcripts_meta)} toggled transcripts in 'some' mode (out of {len(all_files_meta)} total)")
                if matched_keys:
                    logger.info(f"[DEBUG] Matched keys: {matched_keys}")
                else:
                    logger.warning(f"[DEBUG] NO MATCHES! Check if toggle state keys match S3 keys")

        elif transcript_listen_mode == 'all':
            # Read all transcripts from event folder
            transcript_prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/events/{event_id}/transcripts/"
            all_files_meta = list_s3_objects_metadata(transcript_prefix)
            relevant_transcripts_meta = [
                f for f in all_files_meta
                if not os.path.basename(f['Key']).startswith('rolling-') and f['Key'].endswith('.txt')
            ]
            logger.info(f"Found {len(relevant_transcripts_meta)} transcripts in 'all' mode")

        # Load and combine transcripts
        if relevant_transcripts_meta:
            # Sort by date
            relevant_transcripts_meta.sort(key=lambda x: x.get('LastModified', ''))

            # Collect all transcript contents
            transcript_contents = []
            for meta in relevant_transcripts_meta:
                key = meta.get('Key')
                name = os.path.basename(key) if key else 'unknown'
                if key:
                    content = read_file_content(key, f"transcript {name}")
                    if content:
                        # Standardized header format
                        transcript_contents.append(
                            f"=== SOURCE: Transcript - {name} (Event: {event_id}) ===\n"
                            f"{content}\n"
                            f"=== END SOURCE: Transcript - {name} ==="
                        )

            if transcript_contents:
                combined = "\n\n".join(transcript_contents)
                transcript_parts.append(f"=== EVENT {event_id} TRANSCRIPTS ===\n{combined}\n=== END EVENT {event_id} TRANSCRIPTS ===")
                logger.info(f"Loaded {len(transcript_contents)} event transcript(s): {len(combined)} chars")
        else:
            logger.warning(f"No transcript content available for event {event_id} in '{transcript_listen_mode}' mode")

    # 2. Get group transcripts if enabled (mirrors api_server.py:5713-5824)
    if event_id == '0000' and groups_read_mode != 'none':
        # Use event access profile to get allowed events (consistent with main chat agent)
        # Fallback to empty set if event profile not provided
        if allowed_events is None:
            logger.warning("Event access profile not provided for groups_read_mode, using fallback (empty)")
            allowed_events = {"0000"}
        if event_types_map is None:
            event_types_map = {}
        if event_profile is None:
            event_profile = {}

        # Get tier 3 (group) events from allowed_events
        tier3_allow_events = {
            ev for ev in allowed_events
            if ev != '0000' and event_types_map.get(ev, 'group').lower() == 'group'
        }

        if tier3_allow_events:
            if groups_read_mode == 'latest':
                logger.info(f"Groups read mode 'latest': fetching from {len(tier3_allow_events)} group events")
                multi_content, success = read_new_transcript_content_multi(agent_name, list(tier3_allow_events))
                if success and multi_content:
                    transcript_parts.append(f"=== GROUP EVENTS LATEST TRANSCRIPTS ===\n{multi_content}\n=== END GROUP EVENTS LATEST TRANSCRIPTS ===")
                    logger.info(f"Loaded group transcripts: {len(multi_content)} chars")

            elif groups_read_mode == 'all':
                logger.info(f"Groups read mode 'all': fetching all transcripts from {len(tier3_allow_events)} group events")
                groups_contents = []
                for gid in tier3_allow_events:
                    all_content = read_all_transcripts_in_folder(agent_name, gid)
                    if all_content:
                        # Standardized header format
                        groups_contents.append(
                            f"=== SOURCE: Group Event - {gid} (Type: group) ===\n"
                            f"{all_content}\n"
                            f"=== END SOURCE: Group Event - {gid} ==="
                        )
                if groups_contents:
                    combined = "\n\n--- EVENT SEPARATOR ---\n\n".join(groups_contents)
                    transcript_parts.append(f"=== ALL GROUP EVENTS TRANSCRIPTS ===\n{combined}\n=== END ALL GROUP EVENTS TRANSCRIPTS ===")
                    logger.info(f"Loaded all group transcripts: {len(combined)} chars")

            elif groups_read_mode == 'breakout':
                # Read all transcripts from breakout events only (excluding visibility_hidden=true)
                # MATCHES api_server.py:5809-5813 exactly
                breakout_event_ids = [
                    ev for ev in allowed_events
                    if ev != '0000'
                    and event_types_map.get(ev) == 'breakout'
                    and not event_profile.get('event_metadata', {}).get(ev, {}).get('visibility_hidden', False)
                ]
                logger.info(f"Groups read mode 'breakout': fetching all transcripts from {len(breakout_event_ids)} breakout events")
                breakout_contents = []
                for breakout_event_id in breakout_event_ids:
                    breakout_transcripts = read_all_transcripts_in_folder(agent_name, breakout_event_id)
                    if breakout_transcripts:
                        # Standardized header format
                        breakout_contents.append(
                            f"=== SOURCE: Group Event - {breakout_event_id} (Type: breakout) ===\n"
                            f"{breakout_transcripts}\n"
                            f"=== END SOURCE: Group Event - {breakout_event_id} ==="
                        )
                if breakout_contents:
                    breakout_block = "\n\n--- EVENT SEPARATOR ---\n\n".join(breakout_contents)
                    transcript_parts.append(f"=== BREAKOUT EVENTS TRANSCRIPTS ===\n{breakout_block}\n=== END BREAKOUT EVENTS TRANSCRIPTS ===")
                    logger.info(f"Loaded breakout transcripts: {len(breakout_block)} chars")
        else:
            logger.info(f"No group events found in allowed_events for groups_read_mode={groups_read_mode}")

    if not transcript_parts:
        logger.warning("No transcript content available for analysis")
        return None

    combined_content = "\n\n".join(transcript_parts)
    logger.info(f"Total transcript content for analysis: {len(combined_content)} chars")
    return combined_content


def get_next_sequence_number(agent_name: str, event_id: str, mode: str, date_str: str, folder: str = 'mlp-previous') -> int:
    """
    Get the next available sequence number for a given date.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: Analysis mode (mirror/lens/portal)
        date_str: Date string in YYYYMMDD format
        folder: Folder to check ('mlp-previous' or 'mlp-history')

    Returns:
        Next available sequence number (e.g., 1, 2, 3...)
    """
    s3_client = get_s3_client()
    if not s3_client:
        logger.warning("S3 client unavailable, defaulting to sequence 1")
        return 1

    prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/{folder}/{event_id}_{mode}_{date_str}_"

    try:
        response = s3_client.list_objects_v2(Bucket=CANVAS_ANALYSIS_BUCKET, Prefix=prefix)

        if 'Contents' not in response:
            return 1

        # Extract sequence numbers from existing files
        sequence_numbers = []
        for obj in response['Contents']:
            filename = os.path.basename(obj['Key'])
            # Format: 0000_mirror_20251015_001.md
            if filename.endswith('.md'):
                parts = filename.replace('.md', '').split('_')
                if len(parts) >= 4:
                    try:
                        seq = int(parts[-1])
                        sequence_numbers.append(seq)
                    except ValueError:
                        continue

        if not sequence_numbers:
            return 1

        # Return max + 1
        return max(sequence_numbers) + 1

    except Exception as e:
        logger.error(f"Error getting next sequence number: {e}", exc_info=True)
        return 1


def get_s3_analysis_doc_key(agent_name: str, event_id: str, mode: str, version: str = 'latest', sequence: Optional[int] = None) -> str:
    """
    Get S3 key for storing analysis document per agent.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: Analysis mode (mirror/lens/portal)
        version: 'latest' | 'previous' | 'YYYYMMDD' (for history) | 'YYYYMMDD_NNN' (for specific version)
        sequence: Optional sequence number (auto-determined if None for previous/history)

    Returns:
        S3 key path for the analysis document

    Format:
        latest:   organizations/{org}/agents/{agent_name}/_canvas/mlp/mlp-latest/{event}_{mode}.md
        previous: organizations/{org}/agents/{agent_name}/_canvas/mlp/mlp-previous/{event}_{mode}_{date}_{seq}.md
        history:  organizations/{org}/agents/{agent_name}/_canvas/mlp/mlp-history/{event}_{mode}_{date}_{seq}.md
    """
    if version == 'latest':
        return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-latest/{event_id}_{mode}.md"
    elif version == 'previous':
        # For previous, we include a date suffix and sequence number
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        if sequence is None:
            sequence = get_next_sequence_number(agent_name, event_id, mode, date_str, folder='mlp-previous')
        seq_str = f"{sequence:03d}"  # Zero-padded 3 digits (001, 002, etc.)
        return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-previous/{event_id}_{mode}_{date_str}_{seq_str}.md"
    else:
        # version is a date string YYYYMMDD or YYYYMMDD_NNN for history
        # If version contains underscore, it includes sequence number
        if '_' in version:
            # Format: YYYYMMDD_NNN
            return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-history/{event_id}_{mode}_{version}.md"
        else:
            # Format: YYYYMMDD - need to find next sequence
            if sequence is None:
                sequence = get_next_sequence_number(agent_name, event_id, mode, version, folder='mlp-history')
            seq_str = f"{sequence:03d}"
            return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-history/{event_id}_{mode}_{version}_{seq_str}.md"


def load_analysis_doc_from_s3(agent_name: str, event_id: str, mode: str, version: str = 'latest') -> Optional[str]:
    """
    Load analysis document from S3.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: Analysis mode (mirror/lens/portal)
        version: 'latest' | 'previous' | 'YYYYMMDD_NNN' (for specific history)

    Returns:
        Markdown content string or None if not found
    """
    try:
        s3_client = get_s3_client()

        # Return None if S3 client is not available
        if not s3_client:
            logger.warning(f"S3 client unavailable, cannot load {version} {mode} analysis for {agent_name}/{event_id}")
            return None

        # Special handling for 'previous' - load most recent previous file
        if version == 'previous':
            previous_prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-previous/{event_id}_{mode}_"
            response = s3_client.list_objects_v2(Bucket=CANVAS_ANALYSIS_BUCKET, Prefix=previous_prefix)

            if 'Contents' not in response or not response['Contents']:
                logger.info(f"No previous {mode} analysis found for {agent_name}/{event_id}")
                return None

            # Sort by last modified to get most recent
            sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            key = sorted_files[0]['Key']
        else:
            # For 'latest' or specific version, use direct key
            key = get_s3_analysis_doc_key(agent_name, event_id, mode, version=version)

        response = s3_client.get_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=key)
        content = response['Body'].read().decode('utf-8')

        logger.info(f"Loaded {version} {mode} analysis from S3: {key}")
        return content
    except AttributeError as e:
        # Handle case where s3_client is None but wasn't caught above
        logger.warning(f"S3 client error: {e}")
        return None
    except Exception as e:
        # Handle NoSuchKey and other S3 errors
        if hasattr(e, '__class__') and 'NoSuchKey' in str(e.__class__.__name__):
            logger.info(f"No S3 {version} analysis document found for {agent_name}/{event_id}/{mode}")
        else:
            logger.error(f"Error loading analysis from S3: {e}", exc_info=True)
        return None


def delete_previous_analysis(agent_name: str, event_id: str, mode: str) -> bool:
    """
    Delete all previous analysis documents from S3.
    Used when starting a new meeting/session to clear old context.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: Analysis mode (mirror/lens/portal)

    Returns:
        True if deleted successfully or didn't exist, False on error
    """
    try:
        s3_client = get_s3_client()

        # Delete all files in mlp-previous folder for this mode
        previous_prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-previous/{event_id}_{mode}_"

        try:
            response = s3_client.list_objects_v2(Bucket=CANVAS_ANALYSIS_BUCKET, Prefix=previous_prefix)

            if 'Contents' in response:
                deleted_count = 0
                for obj in response['Contents']:
                    s3_client.delete_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=obj['Key'])
                    deleted_count += 1
                    logger.info(f"Deleted previous {mode} analysis: {obj['Key']}")
                logger.info(f"Deleted {deleted_count} previous {mode} analysis file(s)")
            else:
                logger.info(f"No previous {mode} analysis files to delete")

        except s3_client.exceptions.NoSuchKey:
            logger.info(f"No previous {mode} analysis to delete")

        return True
    except Exception as e:
        logger.error(f"Error deleting previous analysis: {e}", exc_info=True)
        return False


def save_analysis_doc_to_s3(agent_name: str, event_id: str, mode: str, content: str) -> bool:
    """
    Save analysis document to S3 as markdown.
    Implements versioning flow: latest → previous (with date) → history (if previous exists)

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: Analysis mode (mirror/lens/portal)
        content: Analysis document markdown content

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        s3_client = get_s3_client()
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')

        latest_key = get_s3_analysis_doc_key(agent_name, event_id, mode, version='latest')
        previous_key = get_s3_analysis_doc_key(agent_name, event_id, mode, version='previous')

        # Step 1: If previous exists, move it to history (copy then delete)
        try:
            # First, list all previous versions to find the most recent one
            previous_prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-previous/{event_id}_{mode}_"
            previous_response = s3_client.list_objects_v2(Bucket=CANVAS_ANALYSIS_BUCKET, Prefix=previous_prefix)

            if 'Contents' in previous_response and previous_response['Contents']:
                # Sort by last modified to get the most recent previous
                sorted_previous = sorted(previous_response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                most_recent_previous_key = sorted_previous[0]['Key']

                # Extract date and sequence from filename
                # Format: 0000_mirror_20251015_001.md
                previous_filename = os.path.basename(most_recent_previous_key)
                parts = previous_filename.replace('.md', '').split('_')

                if len(parts) >= 4:
                    previous_date = parts[-2]  # YYYYMMDD
                    previous_seq = parts[-1]   # 001
                    history_version = f"{previous_date}_{previous_seq}"
                    history_key = get_s3_analysis_doc_key(agent_name, event_id, mode, version=history_version)

                    # Copy previous to history
                    s3_client.copy_object(
                        Bucket=CANVAS_ANALYSIS_BUCKET,
                        CopySource={'Bucket': CANVAS_ANALYSIS_BUCKET, 'Key': most_recent_previous_key},
                        Key=history_key
                    )
                    logger.info(f"Copied previous {mode} analysis to history: {history_key}")

                    # Delete from previous folder after successful copy
                    s3_client.delete_object(
                        Bucket=CANVAS_ANALYSIS_BUCKET,
                        Key=most_recent_previous_key
                    )
                    logger.info(f"Deleted previous {mode} analysis from previous folder: {most_recent_previous_key}")
                else:
                    logger.warning(f"Could not parse previous filename: {previous_filename}")
            else:
                logger.info(f"No existing previous {mode} analysis to archive to history")

        except s3_client.exceptions.NoSuchKey:
            logger.info(f"No existing previous {mode} analysis to archive to history")
        except Exception as archive_err:
            logger.warning(f"Could not archive previous to history: {archive_err}")

        # Step 2: Move current latest to previous (with today's date in metadata)
        try:
            latest_obj = s3_client.head_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=latest_key)

            # Copy latest to previous
            s3_client.copy_object(
                Bucket=CANVAS_ANALYSIS_BUCKET,
                CopySource={'Bucket': CANVAS_ANALYSIS_BUCKET, 'Key': latest_key},
                Key=previous_key,
                Metadata={
                    'date': date_str,
                    'mode': mode,
                    'agent': agent_name,
                    'event': event_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                MetadataDirective='REPLACE'
            )
            logger.info(f"Moved latest {mode} analysis to previous: {previous_key}")
        except s3_client.exceptions.NoSuchKey:
            logger.info(f"No existing latest {mode} analysis to move to previous")
        except Exception as move_err:
            logger.warning(f"Could not move latest to previous: {move_err}")

        # Step 3: Save new content as latest
        s3_client.put_object(
            Bucket=CANVAS_ANALYSIS_BUCKET,
            Key=latest_key,
            Body=content.encode('utf-8'),
            ContentType='text/markdown',
            Metadata={
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'date': date_str,
                'mode': mode,
                'agent': agent_name,
                'event': event_id
            }
        )

        logger.info(f"Saved {mode} analysis to latest: {latest_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving analysis to S3: {e}", exc_info=True)
        return False


def list_canvas_docs(agent_name: str) -> List[Dict[str, Any]]:
    """
    List all documents in _canvas/docs/ folder.

    Args:
        agent_name: Agent name

    Returns:
        List of dict with keys: 'key', 'filename', 'size', 'last_modified'
    """
    s3_client = get_s3_client()
    if not s3_client:
        logger.error("S3 client unavailable for listing canvas docs")
        return []

    prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/docs/"
    docs = []

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=CANVAS_ANALYSIS_BUCKET, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Skip the folder itself
                    if obj['Key'] == prefix or obj['Key'].endswith('/'):
                        continue

                    docs.append({
                        'key': obj['Key'],
                        'filename': os.path.basename(obj['Key']),
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })

        logger.info(f"Found {len(docs)} documents in canvas docs folder for {agent_name}")
        return docs

    except Exception as e:
        logger.error(f"Error listing canvas docs for {agent_name}: {e}", exc_info=True)
        return []


def read_canvas_doc(s3_key: str) -> Optional[str]:
    """
    Read a document from _canvas/docs/.

    Args:
        s3_key: Full S3 key path to the document

    Returns:
        Document content as string, or None if error

    Note:
        Currently supports text/markdown files. PDF extraction is TODO.
    """
    s3_client = get_s3_client()
    if not s3_client:
        logger.error("S3 client unavailable for reading canvas doc")
        return None

    try:
        response = s3_client.get_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=s3_key)

        # Get content type
        content_type = response.get('ContentType', '')
        filename = os.path.basename(s3_key)

        # Handle different file types
        if 'text' in content_type or 'markdown' in content_type or s3_key.endswith(('.txt', '.md')):
            content = response['Body'].read().decode('utf-8')
            logger.info(f"Read text document: {filename} ({len(content)} chars)")
            return content

        elif 'pdf' in content_type or s3_key.endswith('.pdf'):
            # TODO: Add PDF extraction (could use PyPDF2, pdfplumber, or similar)
            logger.warning(f"PDF extraction not yet implemented for {filename}")
            return f"[PDF Document: {filename}]\n(PDF content extraction not yet implemented)"

        else:
            # Unsupported or binary file
            logger.warning(f"Unsupported content type '{content_type}' for {filename}")
            return f"[Binary Document: {filename}]\n(Content type: {content_type})"

    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Canvas doc not found: {s3_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading canvas doc {s3_key}: {e}", exc_info=True)
        return None


def get_all_canvas_source_content(
    agent_name: str,
    event_id: str,
    transcript_listen_mode: str = 'latest',
    groups_read_mode: str = 'none',
    individual_raw_transcript_toggle_states: Optional[Dict[str, bool]] = None,
    saved_transcript_memory_mode: str = 'none',
    individual_memory_toggle_states: Optional[Dict[str, bool]] = None,
    saved_transcript_groups_mode: str = 'none',
    include_memorized_summaries: bool = True,
    allowed_events: Optional[set] = None,
    event_types_map: Optional[Dict[str, str]] = None,
    event_profile: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get all source content for canvas/MLP agents:
    1. Transcripts (based on settings toggles)
    2. Additional docs from _canvas/docs/

    Args:
        agent_name: Agent name
        event_id: Event ID
        transcript_listen_mode: 'none' | 'latest' | 'some' | 'all'
        groups_read_mode: 'none' | 'latest' | 'all' | 'breakout'
        individual_raw_transcript_toggle_states: Dict of s3Key -> bool for "some" mode
        saved_transcript_memory_mode: 'none' | 'some' | 'all' for memorized transcripts
        individual_memory_toggle_states: Dict for memorized transcript "some" mode
        include_memorized_summaries: Include memorized transcript summaries in output
        allowed_events: Set of event IDs user has access to (from get_event_access_profile)
        event_types_map: Dict mapping event_id -> event_type (from get_event_access_profile)
        event_profile: Full event profile dict (includes event_metadata for visibility_hidden check)

    Returns:
        Combined source content with standardized headers
    """
    parts = []

    # 1. Get transcripts (existing function with standardized headers)
    transcript_content = get_transcript_content_for_analysis(
        agent_name=agent_name,
        event_id=event_id,
        transcript_listen_mode=transcript_listen_mode,
        groups_read_mode=groups_read_mode,
        individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states,
        allowed_events=allowed_events,
        event_types_map=event_types_map,
        event_profile=event_profile
    )

    if transcript_content:
        parts.append(transcript_content)
        logger.info(f"Loaded transcript content: {len(transcript_content)} chars")

    # 2. Get additional canvas docs
    canvas_docs = list_canvas_docs(agent_name)

    if canvas_docs:
        docs_parts = []
        for doc in canvas_docs:
            content = read_canvas_doc(doc['key'])
            if content:
                # Determine file type for header
                filename = doc['filename']
                file_ext = os.path.splitext(filename)[1].lower()
                file_type = {
                    '.pdf': 'PDF',
                    '.txt': 'Text',
                    '.md': 'Markdown',
                    '.doc': 'Word',
                    '.docx': 'Word'
                }.get(file_ext, 'Document')

                size_kb = doc['size'] // 1024

                # Standardized header format
                docs_parts.append(
                    f"=== SOURCE: Document - {filename} (Type: {file_type}, Size: {size_kb}KB) ===\n"
                    f"{content}\n"
                    f"=== END SOURCE: Document - {filename} ==="
                )

        if docs_parts:
            combined_docs = "\n\n".join(docs_parts)
            parts.append(f"=== ADDITIONAL DOCUMENTS ===\n{combined_docs}\n=== END ADDITIONAL DOCUMENTS ===")
            logger.info(f"Loaded {len(canvas_docs)} additional documents: {len(combined_docs)} chars")

    # 3. Memorized transcript summaries
    if include_memorized_summaries:
        summaries = get_memorized_transcript_summaries(
            agent_name=agent_name,
            event_id=event_id,
            saved_transcript_memory_mode=saved_transcript_memory_mode,
            individual_memory_toggle_states=individual_memory_toggle_states,
            allowed_events=allowed_events,
            event_profile=event_profile,
            event_types_map=event_types_map,
            groups_mode=saved_transcript_groups_mode,
        )

        if summaries:
            summaries_block = format_memorized_transcripts_block(summaries)
            parts.append(summaries_block)
            logger.info(
                "Loaded memorized transcript summaries for %s/%s: %d summaries (%d chars)",
                agent_name,
                event_id,
                len(summaries),
                len(summaries_block),
            )

    if not parts:
        logger.warning("No source content available (no transcripts, no canvas docs)")
        return ""

    combined = "\n\n".join(parts)
    logger.info(f"Total canvas source content: {len(combined)} chars")
    return combined


def generate_analysis_metadata_header(
    agent_name: str,
    event_id: str,
    mode: str,
    transcript_listen_mode: str,
    groups_read_mode: str,
    source_content: str,
    allowed_events: Optional[set] = None,
    event_types_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate metadata header for MLP analysis documents showing sources used.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: Analysis mode (mirror/lens/portal)
        transcript_listen_mode: Transcript listen mode setting
        groups_read_mode: Groups read mode setting
        source_content: The actual source content that was analyzed
        allowed_events: Set of allowed events (for counting group/breakout events)
        event_types_map: Event types map (for identifying breakout events)

    Returns:
        Markdown header string
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    # Count sources from the source_content string by looking for source markers
    transcript_count = source_content.count('=== SOURCE: Transcript -')
    group_event_count = source_content.count('=== SOURCE: Group Event -')
    document_count = source_content.count('=== SOURCE: Document -')

    # Identify breakout events if in breakout mode
    breakout_events = []
    if groups_read_mode == 'breakout' and allowed_events and event_types_map:
        breakout_events = [
            ev for ev in allowed_events
            if ev != '0000' and event_types_map.get(ev) == 'breakout'
        ]

    header = f"""# Analysis Metadata

**Agent:** {agent_name}
**Event:** {event_id}
**Mode:** {mode.upper()}
**Generated:** {timestamp}

## Sources Analyzed

**Transcript Settings:**
- Listen Mode: `{transcript_listen_mode}`
- Groups Mode: `{groups_read_mode}`

**Sources Loaded:**
- Individual Transcripts: {transcript_count}
- Group Events: {group_event_count}
- Additional Documents: {document_count}
"""

    if breakout_events:
        header += f"\n**Breakout Events Included:** {', '.join(sorted(breakout_events))}\n"

    header += "\n---\n\n"

    return header


def run_analysis_agent(
    agent_name: str,
    event_id: str,
    mode: str,
    transcript_content: str,
    event_type: str = 'shared',
    personal_layer: Optional[str] = None,
    personal_event_id: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    enable_rag: bool = True
) -> Optional[str]:
    """
    Run a single analysis agent to generate mode-specific analysis document.
    Uses Groq with gpt-oss-120b model.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: One of 'mirror', 'lens', 'portal'
        transcript_content: Transcript(s) to analyze
        event_type: Event type
        personal_layer: Personal agent layer
        personal_event_id: Personal event ID
        groq_api_key: Groq API key (defaults to env)
        enable_rag: If True, retrieve context from Pinecone (default: True)

    Returns:
        Analysis document markdown or None on error
    """
    logger.info(f"Running {mode} analysis agent for {agent_name}/{event_id} (RAG: {enable_rag})")

    # RAG retrieval for MLP agents
    rag_context = ""
    if enable_rag:
        try:
            # Create summary of source content for RAG query
            summary = transcript_content[:500] if len(transcript_content) > 500 else transcript_content
            rag_query = f"Context for analyzing: {summary}"

            retriever = RetrievalHandler(
                index_name="river",
                agent_name=agent_name,
                event_id=event_id,
                anthropic_api_key=get_api_key(agent_name, 'anthropic'),
                openai_api_key=get_api_key(agent_name, 'openai'),
                event_type=event_type,
                personal_event_id=personal_event_id,
                include_t3=(event_id == "0000"),
            )

            rag_docs = retriever.get_relevant_context_tiered(
                query=rag_query,
                tier_caps=[4, 8, 6, 6, 4],  # Modest caps for analysis
                include_t3=(event_id == "0000")
            )

            if rag_docs:
                rag_parts = []
                for doc in rag_docs:
                    source_label = doc.metadata.get('source_label', '')
                    filename = doc.metadata.get('filename', 'unknown')
                    tier = doc.metadata.get('retrieval_tier', 'unknown')
                    score = doc.metadata.get('score', 0)

                    rag_parts.append(
                        f"--- Retrieved Context (tier: {tier}, score: {score:.3f}, source: {filename}) ---\n"
                        f"{doc.page_content}\n"
                        f"--- End Retrieved Context ---"
                    )

                rag_context = "\n\n".join(rag_parts)
                logger.info(f"Retrieved {len(rag_docs)} context docs for {mode} analysis")
            else:
                logger.info(f"No RAG context retrieved for {mode} analysis")

        except Exception as rag_err:
            logger.error(f"RAG retrieval failed for {mode} analysis: {rag_err}", exc_info=True)

    # Build analysis prompt
    system_prompt = get_analysis_agent_prompt(
        agent_name=agent_name,
        event_id=event_id,
        mode=mode,
        event_type=event_type,
        personal_layer=personal_layer,
        personal_event_id=personal_event_id,
        transcript_content=transcript_content,
        rag_context=rag_context
    )

    # Get API key (get_api_key already imported at top of file)
    if not groq_api_key:
        groq_api_key = get_api_key(agent_name, 'groq')

    if not groq_api_key:
        logger.error(f"No Groq API key available for analysis agent")
        return None

    # Initialize Groq client
    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
        return None

    # Get rate limiter for concurrent request management
    rate_limiter = get_groq_rate_limiter()

    # Define the Groq API call as a callable for rate limiter
    def make_groq_call():
        model = "openai/gpt-oss-120b"  # Groq model for canvas analysis
        logger.info(f"Calling Groq API for {mode} analysis (model: {model})")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze the source data and produce the {mode} mode analysis document as instructed."}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            stream=False
        )

        # Extract content from response
        analysis_doc = response.choices[0].message.content

        if not analysis_doc:
            logger.error(f"Empty analysis document returned for {mode} mode")
            return None

        logger.info(f"Successfully generated {mode} analysis: {len(analysis_doc)} chars")
        return analysis_doc

    # Execute with rate limiting and retry logic
    result = rate_limiter.execute_with_rate_limit(
        groq_call=make_groq_call,
        context=f"{mode}_analysis_{agent_name}"
    )

    return result


def get_or_generate_analysis_doc(
    agent_name: str,
    event_id: str,
    depth_mode: str,
    force_refresh: bool = False,
    clear_previous: bool = False,
    transcript_listen_mode: str = 'latest',
    groups_read_mode: str = 'none',
    individual_raw_transcript_toggle_states: Optional[Dict[str, bool]] = None,
    saved_transcript_memory_mode: str = 'none',
    individual_memory_toggle_states: Optional[Dict[str, bool]] = None,
    saved_transcript_groups_mode: str = 'none',
    event_type: str = 'shared',
    personal_layer: Optional[str] = None,
    personal_event_id: Optional[str] = None,
    allowed_events: Optional[set] = None,
    event_types_map: Optional[Dict[str, str]] = None,
    event_profile: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get analysis documents (current and previous) for the specified mode, generating if needed.
    Uses hybrid storage: Memory cache → S3 → Generate new

    Args:
        agent_name: Agent name
        event_id: Event ID
        depth_mode: One of 'mirror', 'lens', 'portal'
        force_refresh: If True, bypass cache and regenerate
        clear_previous: If True, delete previous analysis (new meeting/context)
        transcript_listen_mode: Listen mode for event's own transcripts (none/latest/some/all)
        groups_read_mode: Groups read mode from settings (none/latest/all/breakout)
        individual_raw_transcript_toggle_states: Dict of s3Key -> bool for "some" mode filtering
        saved_transcript_memory_mode: Memorized transcript include mode
        individual_memory_toggle_states: Toggle states for memorized transcript "some" mode
        saved_transcript_groups_mode: Groups mode for memorized transcripts (none/latest/all/breakout)
        event_type: Event type
        personal_layer: Personal agent layer
        personal_event_id: Personal event ID
        allowed_events: Set of event IDs user has access to (from get_event_access_profile)
        event_types_map: Dict mapping event_id -> event_type (from get_event_access_profile)
        event_profile: Full event profile dict (includes event_metadata for visibility_hidden check)

    Returns:
        Tuple of (current_doc, previous_doc) - either can be None
    """
    cache_key = f"{agent_name}_{event_id}_{depth_mode}"

    # Delete previous analysis if requested (new meeting/session context)
    if clear_previous:
        logger.info(f"Clearing previous {depth_mode} analysis for {agent_name}/{event_id} (new context)")
        delete_previous_analysis(agent_name, event_id, depth_mode)
        # Also clear from memory cache
        if cache_key in CANVAS_ANALYSIS_CACHE:
            CANVAS_ANALYSIS_CACHE[cache_key]['previous'] = None

    # 1. Check memory cache (unless forced refresh)
    if not force_refresh and cache_key in CANVAS_ANALYSIS_CACHE:
        cached = CANVAS_ANALYSIS_CACHE[cache_key]
        age_seconds = (datetime.now(timezone.utc) - cached['timestamp']).total_seconds()
        age_minutes = age_seconds / 60

        if age_minutes < ANALYSIS_CACHE_TTL_MINUTES:
            logger.info(f"Using memory cached {depth_mode} analysis for {agent_name}/{event_id} (age: {age_minutes:.1f}m)")
            return cached['current'], cached.get('previous')
        else:
            logger.info(f"Memory cached {depth_mode} analysis expired (age: {age_minutes:.1f}m)")

    # 2. Check S3 storage (unless forced refresh)
    if not force_refresh:
        s3_current = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, version='latest')
        s3_previous = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, version='previous')

        if s3_current:
            # S3 docs found - restore to memory cache with current timestamp
            logger.info(f"Using S3 cached {depth_mode} analysis for {agent_name}/{event_id}")

            CANVAS_ANALYSIS_CACHE[cache_key] = {
                'current': s3_current,
                'previous': s3_previous,
                'timestamp': datetime.now(timezone.utc),
                'mode': depth_mode,
                'agent': agent_name,
                'event': event_id
            }

            return s3_current, s3_previous

    # 3. Generate new analysis
    logger.info(f"Generating fresh {depth_mode} analysis for {agent_name}/{event_id}")

    # Get ALL source content (transcripts + additional docs) based on Settings > Memory
    all_source_content = get_all_canvas_source_content(
        agent_name=agent_name,
        event_id=event_id,
        transcript_listen_mode=transcript_listen_mode,
        groups_read_mode=groups_read_mode,
        individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states,
        saved_transcript_memory_mode=saved_transcript_memory_mode,
        individual_memory_toggle_states=individual_memory_toggle_states,
        saved_transcript_groups_mode=saved_transcript_groups_mode,
        allowed_events=allowed_events,
        event_types_map=event_types_map,
        event_profile=event_profile
    )

    if not all_source_content:
        logger.warning(f"No source content available for analysis (no transcripts or docs)")
        return None, None

    # Run analysis agent
    new_analysis_doc = run_analysis_agent(
        agent_name=agent_name,
        event_id=event_id,
        mode=depth_mode,
        transcript_content=all_source_content,
        event_type=event_type,
        personal_layer=personal_layer,
        personal_event_id=personal_event_id
    )

    if not new_analysis_doc:
        logger.error(f"Failed to generate {depth_mode} analysis")
        return None, None

    # 3.5. Prepend metadata header showing sources used
    metadata_header = generate_analysis_metadata_header(
        agent_name=agent_name,
        event_id=event_id,
        mode=depth_mode,
        transcript_listen_mode=transcript_listen_mode,
        groups_read_mode=groups_read_mode,
        source_content=all_source_content,
        allowed_events=allowed_events,
        event_types_map=event_types_map
    )
    new_analysis_doc_with_header = metadata_header + new_analysis_doc

    # 4. Save to S3 (this moves current to previous automatically)
    save_analysis_doc_to_s3(agent_name, event_id, depth_mode, new_analysis_doc_with_header)

    # 5. Load the previous that was just created (what was current before)
    previous_doc = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, version='previous')

    # 6. Cache result in memory with both current and previous
    CANVAS_ANALYSIS_CACHE[cache_key] = {
        'current': new_analysis_doc_with_header,
        'previous': previous_doc,
        'timestamp': datetime.now(timezone.utc),
        'mode': depth_mode,
        'agent': agent_name,
        'event': event_id
    }

    logger.info(f"Cached {depth_mode} analysis (current + previous) in memory and S3 for {agent_name}/{event_id}")
    return new_analysis_doc_with_header, previous_doc


def get_analysis_status(agent_name: str, event_id: str, depth_mode: str) -> Dict[str, Any]:
    """
    Get status of analysis document for UI display.
    Checks both memory cache and S3 storage.

    Args:
        agent_name: Agent name
        event_id: Event ID
        depth_mode: One of 'mirror', 'lens', 'portal'

    Returns:
        Status dict with 'state' and optional 'timestamp'
    """
    cache_key = f"{agent_name}_{event_id}_{depth_mode}"

    # Check memory cache first
    if cache_key in CANVAS_ANALYSIS_CACHE:
        cached = CANVAS_ANALYSIS_CACHE[cache_key]
        age_seconds = (datetime.now(timezone.utc) - cached['timestamp']).total_seconds()
        age_minutes = age_seconds / 60

        if age_minutes < ANALYSIS_CACHE_TTL_MINUTES:
            return {
                'state': 'ready',
                'timestamp': cached['timestamp'].isoformat()
            }

    # Check S3 storage
    s3_current = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, version='latest')
    if s3_current:
        # S3 docs exist, report as ready
        return {
            'state': 'ready',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    return {'state': 'none'}


def invalidate_analysis_cache(agent_name: str, event_id: str, depth_mode: Optional[str] = None):
    """
    Invalidate cached analysis documents.

    Args:
        agent_name: Agent name
        event_id: Event ID
        depth_mode: If specified, only invalidate this mode. Otherwise all modes.
    """
    if depth_mode:
        cache_key = f"{agent_name}_{event_id}_{depth_mode}"
        if cache_key in CANVAS_ANALYSIS_CACHE:
            del CANVAS_ANALYSIS_CACHE[cache_key]
            logger.info(f"Invalidated {depth_mode} analysis cache for {agent_name}/{event_id}")
    else:
        # Invalidate all modes
        for mode in ['mirror', 'lens', 'portal']:
            cache_key = f"{agent_name}_{event_id}_{mode}"
            if cache_key in CANVAS_ANALYSIS_CACHE:
                del CANVAS_ANALYSIS_CACHE[cache_key]
        logger.info(f"Invalidated all analysis caches for {agent_name}/{event_id}")
