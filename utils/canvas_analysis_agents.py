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
=== ANALYSIS TASK: MIRROR MODE (EDGES) ===
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document identifying EXPLICIT BUT PERIPHERAL information.

Your role is to surface what IS explicitly stated but sits at the margins of the conversation.

Focus areas - explore each in depth:
- Minority viewpoints explicitly voiced (quote speakers, explain context)
- Side comments and tangential observations (what was said in passing)
- Outlier perspectives mentioned (ideas not in the mainstream flow)
- Peripheral but concrete concerns (worries expressed at the edges)

Structure your analysis as:

# Mirror Analysis: Edge Cases

## Minority Viewpoints
[For EACH minority viewpoint found:
- Quote the exact words used
- Identify who said it (Speaker 1, Speaker 2, etc.)
- Explain the context in which it was raised
- Note why it's peripheral to the main conversation]

## Tangential Observations
[For EACH side comment:
- What was said
- When in the conversation
- How it relates (or doesn't) to the main topic
- Why it's noteworthy despite being off-topic]

## Outlier Concerns
[For EACH peripheral concern:
- The specific concern raised
- Direct quotes supporting it
- Why it's an outlier
- Potential significance despite being peripheral]

## Summary
[Brief synthesis of the edge landscape - what patterns emerge at the periphery?]

Guidelines:
- TARGET LENGTH: 1500-2000 tokens - be thorough and comprehensive
- Use exact quotes liberally with attribution
- Provide context for each item you surface
- Stay grounded in transcript content - no speculation
- Plain markdown only (no special formatting like **, _, etc.)
- If truly no peripheral content exists, explain what was examined and why nothing qualified
- This is a serious analytical document - depth and evidence are valued
=== END ANALYSIS TASK ===
""",

        'lens': """
=== ANALYSIS TASK: LENS MODE (LATENT NEEDS) ===
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document identifying HIDDEN PATTERNS and UNSPOKEN REQUIREMENTS.

Your role is to identify what the conversation implies about deeper contextual needs - reading between the lines with rigor.

Focus areas - explore each systematically:
- Recurring themes across different speakers and moments
- Emotional undercurrents and group dynamics (tension, avoidance, energy shifts)
- Systemic issues beneath surface symptoms (root causes, not just effects)
- Paradoxes and contradictions (where words and reality diverge)
- What's being avoided, protected, or left unsaid

Structure your analysis as:

# Lens Analysis: Latent Needs

## Hidden Patterns
[For EACH pattern identified:
- Describe the pattern clearly
- Cite 2-3 specific examples from transcript with quotes
- Explain what makes this pattern significant
- Connect to broader themes or needs]

## Emotional Dynamics
[For EACH dynamic observed:
- Name the dynamic (e.g., "tension around resource allocation")
- Evidence from tone, word choice, silences, repetition
- Who's involved and how they're positioned
- What this reveals about unspoken concerns]

## Systemic Issues
[For EACH systemic issue:
- The surface symptom being discussed
- The deeper root cause you're inferring
- Evidence trail from transcript
- Why treating symptoms alone won't work]

## Unspoken Needs
[For EACH latent need:
- What the group actually requires
- How you know (cite behavioral evidence)
- Why it remains unspoken
- What would shift if it were named]

## Paradoxes and Contradictions
[For EACH paradox:
- State the contradiction clearly
- Quote evidence of both sides
- Analyze what this tension reveals
- Potential resolution paths]

## Synthesis
[Meta-analysis: What's the deeper story this conversation is telling? What need is most central?]

Guidelines:
- TARGET LENGTH: 1500-2000 tokens - depth over brevity
- Every inference must be evidenced from transcript
- Surface paradoxes explicitly - they're goldmines
- Use questioning analytical language ("seems to suggest", "may indicate")
- Plain markdown only (no **, _, etc.)
- Focus on what's implied, not what's stated directly
- This is interpretive but rigorous - show your analytical work
=== END ANALYSIS TASK ===
""",

        'portal': """
=== ANALYSIS TASK: PORTAL MODE (EMERGENT QUESTIONS) ===
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document framing EMERGENT POSSIBILITIES AS QUESTIONS.

Your role is to open possibility spaces by formulating transformative questions derived from lens-level patterns and paradoxes.

Core principles:
- Every question must be traceable to evidence (quote the pattern it emerges from)
- Questions challenge assumptions, not people
- Frame possibilities as invitations to explore, never prescriptions
- Questions should be generative - they open space rather than close it

Focus areas - develop each fully:
- Questions that challenge limiting assumptions embedded in the conversation
- Questions that identify transformation opportunities from latent needs
- Questions about paradigm shifts suggested by contradictions
- Questions predicting intervention outcomes based on patterns
- Questions that identify leverage points and opportunity windows

Structure your analysis as:

# Portal Analysis: Emergent Questions

## Transformative Questions
[For EACH transformative question (aim for 3-5):
- State the question clearly
- Quote the lens-level pattern or paradox it emerges from
- Explain what shift in thinking this question invites
- Note what becomes possible if this question is engaged]

## Assumption Challenges
[For EACH assumption-challenging question (aim for 3-5):
- State the question
- Identify the hidden assumption being questioned (with transcript evidence)
- Explain why this assumption may be limiting
- Describe the expanded possibility space if the assumption is released]

## Intervention Pathways
[For EACH "what if we..." question (aim for 3-5):
- State the question as an intervention possibility
- Connect to latent needs identified in lens analysis
- Cite transcript evidence for why this pathway could matter
- Explore potential outcomes or ripple effects]

## Opportunity Windows
[For EACH leverage-point question (aim for 3-5):
- State the question identifying the leverage point
- Explain why this moment/issue is strategic
- Quote evidence of readiness or opening
- Describe what could shift if this window is engaged]

## Synthesis Questions
[2-3 meta-level questions that integrate across the analysis:
- What's the most generative question this conversation wants to ask?
- Where's the greatest potential for transformation?
- What question, if truly engaged, could shift everything?]

Guidelines:
- TARGET LENGTH: 1500-2000 tokens - develop questions with depth
- Each question needs its evidence trail from transcript
- Quality over quantity - better 15 well-developed questions than 30 shallow ones
- Use probing, open-ended language ("How might...", "What if...", "What would it mean to...")
- Plain markdown only (no **, _, etc.)
- Questions should invite genuine exploration, not lead to predetermined answers
- This is strategic inquiry - the questions themselves are interventions
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
    groups_read_mode: str = 'none'
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
                        transcript_contents.append(f"--- Transcript: {name} ---\n{content}\n--- End Transcript: {name} ---")

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
                            transcript_parts.append(f"=== GROUP EVENTS TRANSCRIPTS ===\n{multi_content}\n=== END GROUP EVENTS TRANSCRIPTS ===")
                            logger.info(f"Loaded group transcripts: {len(multi_content)} chars")

                    elif groups_read_mode == 'all':
                        logger.info(f"Groups read mode 'all': fetching all transcripts from {len(tier3_allow_events)} group events")
                        groups_contents = []
                        for gid in tier3_allow_events:
                            all_content = read_all_transcripts_in_folder(agent_name, gid)
                            if all_content:
                                groups_contents.append(f"--- Group {gid} All Transcripts ---\n{all_content}")
                        if groups_contents:
                            combined = "\n\n".join(groups_contents)
                            transcript_parts.append(f"=== ALL GROUP TRANSCRIPTS ===\n{combined}\n=== END ALL GROUP TRANSCRIPTS ===")
                            logger.info(f"Loaded all group transcripts: {len(combined)} chars")

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
        groups_read_mode=groups_read_mode
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
