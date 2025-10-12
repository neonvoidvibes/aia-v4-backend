"""
Canvas Analysis Agents - Generate pre-analyzed insights for canvas modes.

This module runs specialized analysis agents that:
1. Use full main chat agent taxonomy (all context layers)
2. Read selected transcripts based on Settings > Memory
3. Produce markdown analysis documents for mirror/lens/portal modes
4. Store docs in S3 (./agents/_canvas/docs/) + cache for fast responses
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from groq import Groq

from .prompt_builder import prompt_builder
from .s3_utils import get_s3_client
from .transcript_utils import get_latest_transcript_file, read_all_transcripts_in_folder

logger = logging.getLogger(__name__)

# In-memory cache for analysis documents
# Structure: {cache_key: {'current': str, 'previous': str|None, 'timestamp': datetime, ...}}
CANVAS_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}

# Cache TTL in minutes
ANALYSIS_CACHE_TTL_MINUTES = 15

# S3 storage configuration
CANVAS_ANALYSIS_BUCKET = os.getenv("S3_BUCKET_NAME", "aiademomagicaudio")
CANVAS_ANALYSIS_ORG = os.getenv("S3_ORGANIZATION", "river")


def get_analysis_agent_prompt(
    agent_name: str,
    event_id: str,
    mode: str,
    event_type: str,
    personal_layer: Optional[str],
    personal_event_id: Optional[str],
    transcript_content: str
) -> str:
    """
    Build full taxonomy prompt + mode-specific analysis instructions.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: One of 'mirror', 'lens', 'portal'
        event_type: Event type (personal/group/shared)
        personal_layer: Personal agent layer content
        personal_event_id: Personal event ID
        transcript_content: Transcript(s) to analyze

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

    # 2. Add transcript content
    transcript_section = f"""
=== TRANSCRIPT DATA ===
{transcript_content}
=== END TRANSCRIPT DATA ===
"""

    # 3. Add mode-specific analysis task
    analysis_tasks = {
        'mirror': """
=== ANALYSIS TASK: MIRROR MODE (EXPLICIT INFORMATION) ===
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document reflecting EXPLICIT INFORMATION at both surface and depth levels.

Your role: Mirror what IS explicitly stated - both the obvious center and the peripheral edges.

Tone: Neutral, observational, present tense, no interpretation. Use participants' exact words.

Structure your analysis as:

# Mirror Analysis: Explicit Information

## Surface Level: Center of Distribution (The Obvious)
Reflect what's at the center - the concrete themes directly and repeatedly stated.

[For EACH central theme (aim for 3-5):
- State the theme clearly using participants' exact formulations
- Quote multiple speakers expressing this theme
- Note explicit agreements around this topic
- Identify common denominators in how it's expressed

Language patterns: "You're discussing...", "The group agrees that...", "Several of you have mentioned...", "The main topic is..."]

## Deep Level: Edge Cases (Explicit but Peripheral)
Capture what's explicitly stated but sits at the margins - minority views, side comments, outliers.

[For EACH peripheral item (aim for 3-5):
- Quote the exact words used with attribution
- Explain when and why it's peripheral to the main flow
- Note its explicit nature despite being an outlier
- Assess potential significance despite marginality

Language patterns: "One participant also noted...", "A less discussed but mentioned point...", "While not the main focus, someone raised...", "An interesting side observation was..."]

## Most Pertinent Observation
[Single focused paragraph:
What is the ONE most significant explicit observation from this conversation - whether from center or edges? This should be concrete, grounded in what was actually said, and useful for multiple perspectives the canvas user might bring. Don't over-index for any single interpretation - stay focused and factual.]

Guidelines:
- TARGET LENGTH: 1500-2000 tokens total
- Maintain neutral, observational tone throughout
- Use exact quotes liberally with speaker attribution
- Present tense: "The group is discussing...", "Participants mention..."
- Plain markdown only (no **, _, etc.)
- NO interpretation or reading between lines - that's Lens territory
- Comprehensive coverage: surface should feel recognizable, depth should surface what's easily missed
- This is pure reflection - like holding up a mirror to the conversation
=== END ANALYSIS TASK ===
""",

        'lens': """
=== ANALYSIS TASK: LENS MODE (HIDDEN INFORMATION) ===
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document identifying HIDDEN INFORMATION at both surface and depth levels.

Your role: Identify what's IMPLIED - patterns at the surface, latent needs at the depth.

Tone: Analytical, questioning, surface paradoxes and tensions. Use questioning language.

Structure your analysis as:

# Lens Analysis: Hidden Information

## Surface Level: Pattern Recognition Between Data Points
Identify recurring themes and connections not explicitly stated but clearly present.

[For EACH pattern (aim for 3-5):
- Name the pattern clearly
- Cite 2-3 specific examples from different parts of transcript
- Show how seemingly unrelated comments connect
- Note emotional undercurrents or energy shifts
- Identify group dynamics emerging

Language patterns: "There's a pattern emerging around...", "Several comments suggest an underlying...", "The energy shifts when discussing...", "A tension exists between..."]

## Deep Level: Latent Needs Analysis
Surface what the group actually requires contextually - the unspoken needs driving the conversation.

[For EACH latent need (aim for 3-5):
- State the unspoken need clearly
- Cite behavioral evidence from transcript (not just words, but pauses, repetitions, avoidances)
- Identify systemic issues beneath symptoms
- Surface paradoxes and contradictions
- Analyze what's being protected or avoided
- Explain why it remains unspoken
- Predict what would shift if this need were named

Language patterns: "The underlying need seems to be...", "What's not being said directly is...", "The group appears to be protecting...", "A deeper dynamic at play might be..."]

## Most Pertinent Observation
[Single focused paragraph:
What is the ONE most significant hidden pattern or latent need from this conversation? This should illuminate something crucial about the group's contextual requirements while remaining open to multiple interpretive lenses. Don't prescribe a single solution - surface the core dynamic that matters most.]

Guidelines:
- TARGET LENGTH: 1500-2000 tokens total
- Analytical, questioning tone throughout
- Every inference must be evidenced from transcript
- Surface paradoxes explicitly - they're goldmines for Portal
- Use questioning analytical language: "seems to suggest", "may indicate", "appears to be"
- Plain markdown only (no **, _, etc.)
- Focus on what's IMPLIED, not what's explicitly stated - that's Mirror territory
- Systemic thinking: Connect symptoms to root causes
- This is interpretive but rigorous - show your analytical work with evidence trails
=== END ANALYSIS TASK ===
""",

        'portal': """
=== ANALYSIS TASK: PORTAL MODE (EMERGENT QUESTIONS) ===
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document composed ENTIRELY OF QUESTIONS that open possibility spaces.

Your role: Formulate transformative questions - general possibilities at surface, predictive interventions at depth.

Tone: Visionary, possibility-oriented, future tense welcomed. Frame as invitations, not prescriptions.

CRITICAL OUTPUT REQUIREMENT: Your entire output must be QUESTIONS ONLY. No observations, insights, or declarative statements. Every sentence should end with a question mark.

Structure your analysis as:

# Portal Analysis: Emergent Questions

## Surface Level: General Possibilities That Could Transform the System
Formulate questions about transformation opportunities and paradigm shifts.

[For EACH transformative question (aim for 3-5):
- Ask a core transformative question
- Follow with sub-questions that explore what paradigm shift this invites
- Ask questions that challenge limiting assumptions embedded in current thinking
- Pose questions that expand the possibility space

Examples:
"What if you could [X]? How might that shift the current paradigm? What assumptions would need to be released for this to become possible? What new territory would open up?"

Language patterns: "What if you could...", "How might...", "What would it mean to...", "What becomes possible when...", "What if instead of [current approach], you..."]

## Deep Level: Predictive Questions About Specific Interventions
Ask questions about specific actions and their likely outcomes - modeling interventions through inquiry.

[For EACH intervention question set (aim for 3-5):
- Ask: "What if you pursued [X specific action]?"
- Follow with: "What effects might cascade from this?"
- Ask: "What would be the ripple effects across [different parts of system]?"
- Pose: "Who would be affected and how?"
- Question: "What's the highest leverage point for this intervention?"
- Explore: "What transformation pathway might this open?"

Examples:
"What if you invested in [specific action]? What cascading effects might follow? How might this shift relationships between [parties]? What would need to be true for this to succeed? What timeline might be realistic?"

Language patterns: "What if you...", "How might this cascade into...", "What leverage points exist for...", "What would happen if...", "What outcomes might emerge from..."]

## Most Pertinent Question
[Single focused question (can be compound):
What is the ONE most significant question this conversation invites - the question with greatest transformative potential? Make it concrete enough to engage with, yet open enough to invite multiple approaches. This should feel like a strategic doorway, not a prescription. This MUST be phrased as a question.]

Guidelines:
- TARGET LENGTH: 1500-2000 tokens total
- EVERY output must be a QUESTION - no declarative statements, observations, or insights
- Each question set should be grounded in patterns from the transcript
- Use probing, open-ended language: "How might...", "What if...", "What would it mean to...", "What becomes possible when..."
- Plain markdown only (no **, _, etc.)
- Questions should invite genuine exploration and multiple answers, not lead to predetermined conclusions
- Deep level questions should be concrete and predictive - name specific actions and explore outcomes through inquiry
- Quality over quantity: better fewer well-developed question sets than many shallow single questions
- Frame questions as invitations to explore, never as prescriptions disguised as questions
- This is strategic inquiry - the questions themselves are the intervention
=== END ANALYSIS TASK ===
"""
    }

    if mode not in analysis_tasks:
        logger.error(f"Invalid analysis mode: {mode}")
        mode = 'mirror'

    return base_prompt + "\n\n" + transcript_section + "\n\n" + analysis_tasks[mode]


def get_transcript_content_for_analysis(
    agent_name: str,
    event_id: str,
    transcript_listen_mode: str = 'latest',
    groups_read_mode: str = 'none',
    individual_raw_transcript_toggle_states: Optional[Dict[str, bool]] = None
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
                        transcript_contents.append(f"--- START Transcript Source: {name} ---\n{content}\n--- END Transcript Source: {name} ---")

            if transcript_contents:
                combined = "\n\n".join(transcript_contents)
                transcript_parts.append(f"=== EVENT {event_id} TRANSCRIPTS ===\n{combined}\n=== END EVENT {event_id} TRANSCRIPTS ===")
                logger.info(f"Loaded {len(transcript_contents)} event transcript(s): {len(combined)} chars")
        else:
            logger.warning(f"No transcript content available for event {event_id} in '{transcript_listen_mode}' mode")

    # 2. Get group transcripts if enabled (mirrors api_server.py:5713-5746)
    if event_id == '0000' and groups_read_mode != 'none':
        # Get allowed group events from event profile
        from .auth_helpers import get_event_profile
        from .supabase_client import get_supabase_client

        client = get_supabase_client()
        if client:
            try:
                # Get user_id from current context (we'll need to pass this)
                # For now, get all group events for this agent
                event_rows = client.table("events").select("*").eq("agent_name", agent_name).execute()
                tier3_allow_events = {
                    row['event_id'] for row in event_rows.data
                    if row.get('type', 'group').lower() == 'group' and row['event_id'] != '0000'
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
                                groups_contents.append(f"--- Group Event: {gid} ---\n{all_content}")
                        if groups_contents:
                            combined = "\n\n--- EVENT SEPARATOR ---\n\n".join(groups_contents)
                            transcript_parts.append(f"=== ALL GROUP EVENTS TRANSCRIPTS ===\n{combined}\n=== END ALL GROUP EVENTS TRANSCRIPTS ===")
                            logger.info(f"Loaded all group transcripts: {len(combined)} chars")

                    elif groups_read_mode == 'breakout':
                        # Read all transcripts from breakout events only (excluding visibility_hidden=true)
                        # Get breakout events for this agent
                        breakout_rows = client.table("events").select("*").eq("agent_name", agent_name).execute()
                        breakout_event_ids = [
                            row['event_id'] for row in breakout_rows.data
                            if row.get('event_id') != '0000'
                            and row.get('type', '').lower() == 'breakout'
                            and not row.get('visibility_hidden', False)
                        ]
                        logger.info(f"Groups read mode 'breakout': fetching all transcripts from {len(breakout_event_ids)} breakout events")
                        breakout_contents = []
                        for breakout_event_id in breakout_event_ids:
                            breakout_transcripts = read_all_transcripts_in_folder(agent_name, breakout_event_id)
                            if breakout_transcripts:
                                breakout_contents.append(f"--- Breakout Event: {breakout_event_id} ---\n{breakout_transcripts}")
                        if breakout_contents:
                            breakout_block = "\n\n--- EVENT SEPARATOR ---\n\n".join(breakout_contents)
                            transcript_parts.append(f"=== BREAKOUT EVENTS TRANSCRIPTS ===\n{breakout_block}\n=== END BREAKOUT EVENTS TRANSCRIPTS ===")
                            logger.info(f"Loaded breakout transcripts: {len(breakout_block)} chars")

            except Exception as e:
                logger.error(f"Error fetching group transcripts: {e}", exc_info=True)

    if not transcript_parts:
        logger.warning("No transcript content available for analysis")
        return None

    combined_content = "\n\n".join(transcript_parts)
    logger.info(f"Total transcript content for analysis: {len(combined_content)} chars")
    return combined_content


def get_s3_analysis_doc_key(agent_name: str, event_id: str, mode: str, previous: bool = False) -> str:
    """
    Get S3 key for storing analysis document per agent.

    Format: organizations/{org}/agents/{agent_name}/_canvas/{event}_{mode}.md
    Or:     organizations/{org}/agents/{agent_name}/_canvas/{event}_{mode}_previous.md
    """
    suffix = "_previous" if previous else ""
    return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/{event_id}_{mode}{suffix}.md"


def load_analysis_doc_from_s3(agent_name: str, event_id: str, mode: str, previous: bool = False) -> Optional[str]:
    """
    Load analysis document from S3.

    Args:
        agent_name: Agent name
        event_id: Event ID
        mode: Analysis mode (mirror/lens/portal)
        previous: If True, load the _previous version

    Returns:
        Markdown content string or None if not found
    """
    try:
        s3_client = get_s3_client()
        key = get_s3_analysis_doc_key(agent_name, event_id, mode, previous=previous)

        response = s3_client.get_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=key)
        content = response['Body'].read().decode('utf-8')

        version_label = "previous" if previous else "current"
        logger.info(f"Loaded {version_label} {mode} analysis from S3: {key}")
        return content
    except s3_client.exceptions.NoSuchKey:
        version_label = "previous" if previous else "current"
        logger.info(f"No S3 {version_label} analysis document found for {agent_name}/{event_id}/{mode}")
        return None
    except Exception as e:
        logger.error(f"Error loading analysis from S3: {e}", exc_info=True)
        return None


def delete_previous_analysis(agent_name: str, event_id: str, mode: str) -> bool:
    """
    Delete previous analysis document from S3.
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
        previous_key = get_s3_analysis_doc_key(agent_name, event_id, mode, previous=True)

        try:
            s3_client.delete_object(
                Bucket=CANVAS_ANALYSIS_BUCKET,
                Key=previous_key
            )
            logger.info(f"Deleted previous {mode} analysis: {previous_key}")
        except s3_client.exceptions.NoSuchKey:
            logger.info(f"No previous {mode} analysis to delete")

        return True
    except Exception as e:
        logger.error(f"Error deleting previous analysis: {e}", exc_info=True)
        return False


def save_analysis_doc_to_s3(agent_name: str, event_id: str, mode: str, content: str) -> bool:
    """
    Save analysis document to S3 as markdown.
    Moves current version to _previous before saving new version.

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
        current_key = get_s3_analysis_doc_key(agent_name, event_id, mode, previous=False)
        previous_key = get_s3_analysis_doc_key(agent_name, event_id, mode, previous=True)

        # Move current to previous (if current exists)
        try:
            s3_client.copy_object(
                Bucket=CANVAS_ANALYSIS_BUCKET,
                CopySource={'Bucket': CANVAS_ANALYSIS_BUCKET, 'Key': current_key},
                Key=previous_key
            )
            logger.info(f"Moved current {mode} analysis to previous: {previous_key}")
        except s3_client.exceptions.NoSuchKey:
            logger.info(f"No existing current analysis to move to previous for {mode}")
        except Exception as copy_err:
            logger.warning(f"Could not copy current to previous: {copy_err}")

        # Save new current
        s3_client.put_object(
            Bucket=CANVAS_ANALYSIS_BUCKET,
            Key=current_key,
            Body=content.encode('utf-8'),
            ContentType='text/markdown',
            Metadata={
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mode': mode,
                'agent': agent_name,
                'event': event_id
            }
        )

        logger.info(f"Saved {mode} analysis to S3: {current_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving analysis to S3: {e}", exc_info=True)
        return False


def run_analysis_agent(
    agent_name: str,
    event_id: str,
    mode: str,
    transcript_content: str,
    event_type: str = 'shared',
    personal_layer: Optional[str] = None,
    personal_event_id: Optional[str] = None,
    groq_api_key: Optional[str] = None
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

    Returns:
        Analysis document markdown or None on error
    """
    logger.info(f"Running {mode} analysis agent for {agent_name}/{event_id}")

    # Build analysis prompt
    system_prompt = get_analysis_agent_prompt(
        agent_name=agent_name,
        event_id=event_id,
        mode=mode,
        event_type=event_type,
        personal_layer=personal_layer,
        personal_event_id=personal_event_id,
        transcript_content=transcript_content
    )

    # Get API key
    if not groq_api_key:
        from .api_key_manager import get_api_key
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

    # Call Groq LLM for analysis
    try:
        model = "openai/gpt-oss-120b"  # Groq model for canvas analysis
        logger.info(f"Calling Groq API for {mode} analysis (model: {model})")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze the transcript(s) and produce the {mode} mode analysis document as instructed."}
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

    except Exception as e:
        logger.error(f"Error running {mode} analysis agent: {e}", exc_info=True)
        return None


def get_or_generate_analysis_doc(
    agent_name: str,
    event_id: str,
    depth_mode: str,
    force_refresh: bool = False,
    clear_previous: bool = False,
    transcript_listen_mode: str = 'latest',
    groups_read_mode: str = 'none',
    individual_raw_transcript_toggle_states: Optional[Dict[str, bool]] = None,
    event_type: str = 'shared',
    personal_layer: Optional[str] = None,
    personal_event_id: Optional[str] = None
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
        event_type: Event type
        personal_layer: Personal agent layer
        personal_event_id: Personal event ID

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
        s3_current = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, previous=False)
        s3_previous = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, previous=True)

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

    # Get transcript content based on Settings > Memory
    transcript_content = get_transcript_content_for_analysis(
        agent_name=agent_name,
        event_id=event_id,
        transcript_listen_mode=transcript_listen_mode,
        groups_read_mode=groups_read_mode,
        individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states
    )

    if not transcript_content:
        logger.warning(f"No transcript content available for analysis")
        return None, None

    # Run analysis agent
    new_analysis_doc = run_analysis_agent(
        agent_name=agent_name,
        event_id=event_id,
        mode=depth_mode,
        transcript_content=transcript_content,
        event_type=event_type,
        personal_layer=personal_layer,
        personal_event_id=personal_event_id
    )

    if not new_analysis_doc:
        logger.error(f"Failed to generate {depth_mode} analysis")
        return None, None

    # 4. Save to S3 (this moves current to previous automatically)
    save_analysis_doc_to_s3(agent_name, event_id, depth_mode, new_analysis_doc)

    # 5. Load the previous that was just created (what was current before)
    previous_doc = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, previous=True)

    # 6. Cache result in memory with both current and previous
    CANVAS_ANALYSIS_CACHE[cache_key] = {
        'current': new_analysis_doc,
        'previous': previous_doc,
        'timestamp': datetime.now(timezone.utc),
        'mode': depth_mode,
        'agent': agent_name,
        'event': event_id
    }

    logger.info(f"Cached {depth_mode} analysis (current + previous) in memory and S3 for {agent_name}/{event_id}")
    return new_analysis_doc, previous_doc


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
    s3_current = load_analysis_doc_from_s3(agent_name, event_id, depth_mode, previous=False)
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
