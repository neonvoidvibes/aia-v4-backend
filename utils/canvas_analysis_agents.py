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
from typing import Optional, Dict, Any, List
from groq import Groq

from .prompt_builder import prompt_builder
from .s3_utils import get_s3_client
from .transcript_utils import get_latest_transcript_file, read_all_transcripts_in_folder

logger = logging.getLogger(__name__)

# In-memory cache for analysis documents
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
Analyze the transcript(s) and produce a concise markdown document identifying EXPLICIT BUT PERIPHERAL information.

Focus:
- Minority viewpoints explicitly voiced
- Side comments and tangential observations
- Outlier perspectives mentioned
- Peripheral but concrete concerns

Structure your analysis as:

# Mirror Analysis: Edge Cases

## Minority Viewpoints
[Quote and summarize viewpoints from the periphery]

## Tangential Observations
[Side comments and off-topic insights]

## Outlier Concerns
[Less central but explicitly stated concerns]

Rules:
- Maximum 2000 tokens
- Use exact quotes where possible
- Stay grounded in transcript content
- Plain markdown (no special formatting)
- If no peripheral content exists, state that clearly and briefly
=== END ANALYSIS TASK ===
""",

        'lens': """
=== ANALYSIS TASK: LENS MODE (LATENT NEEDS) ===
Analyze the transcript(s) and produce a concise markdown document identifying HIDDEN PATTERNS and UNSPOKEN REQUIREMENTS.

Focus:
- Recurring themes across speakers
- Emotional undercurrents and group dynamics
- Systemic issues beneath surface symptoms
- Paradoxes and contradictions
- What's being avoided or protected

Structure your analysis as:

# Lens Analysis: Latent Needs

## Hidden Patterns
[Recurring themes and connections]

## Emotional Dynamics
[Undercurrents and group tensions]

## Systemic Issues
[Root causes vs symptoms]

## Unspoken Needs
[What the group actually requires but hasn't articulated]

Rules:
- Maximum 2000 tokens
- Cite evidence from transcripts
- Surface paradoxes clearly
- Plain markdown (no special formatting)
- Focus on what's implied, not explicit
=== END ANALYSIS TASK ===
""",

        'portal': """
=== ANALYSIS TASK: PORTAL MODE (EMERGENT QUESTIONS) ===
Analyze the transcript(s) and produce a concise markdown document framing EMERGENT POSSIBILITIES AS QUESTIONS.

Focus:
- Questions that challenge limiting assumptions
- Questions identifying transformation opportunities
- Questions about paradigm shifts
- Questions predicting intervention outcomes
- All questions must be traceable to lens-level patterns

Structure your analysis as:

# Portal Analysis: Emergent Questions

## Transformative Questions
[Questions that could shift the paradigm]

## Assumption Challenges
[Questions that surface hidden beliefs]

## Intervention Pathways
[Questions exploring "what if we..." scenarios]

## Opportunity Windows
[Questions identifying leverage points]

Rules:
- Maximum 2000 tokens
- Every question must reference transcript evidence
- Frame as invitations to explore, not prescriptions
- Plain markdown (no special formatting)
- Derive questions from lens-level insights
=== END ANALYSIS TASK ===
"""
    }

    if mode not in analysis_tasks:
        logger.error(f"Invalid analysis mode: {mode}")
        mode = 'mirror'

    return base_prompt + "\n\n" + transcript_section + "\n\n" + analysis_tasks[mode]


def get_transcript_content_for_analysis(agent_name: str, event_id: str, groups_read_mode: str = 'none') -> Optional[str]:
    """
    Get transcript content based on Settings > Memory configuration.

    This mirrors the main chat agent's transcript reading logic but returns
    the raw content for analysis rather than adding it to system prompt.

    Args:
        agent_name: Agent name
        event_id: Event ID (typically '0000' for canvas)
        groups_read_mode: 'none' | 'latest' | 'all' | 'breakout'

    Returns:
        Combined transcript content or None
    """
    from .transcript_utils import read_new_transcript_content, read_new_transcript_content_multi

    transcript_parts = []

    # 1. Get event's own transcript (latest)
    logger.info(f"Fetching transcript for analysis: agent={agent_name}, event={event_id}")
    content, success = read_new_transcript_content(agent_name, event_id)
    if success and content:
        transcript_parts.append(f"=== EVENT {event_id} TRANSCRIPT ===\n{content}\n=== END EVENT {event_id} TRANSCRIPT ===")
        logger.info(f"Loaded event transcript: {len(content)} chars")
    else:
        logger.warning(f"No transcript content available for event {event_id}")

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


def get_s3_analysis_doc_key(agent_name: str, event_id: str, mode: str) -> str:
    """
    Get S3 key for storing analysis document.

    Format: organizations/{org}/agents/_canvas/docs/{agent}_{event}_{mode}.json
    """
    return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/_canvas/docs/{agent_name}_{event_id}_{mode}.json"


def load_analysis_doc_from_s3(agent_name: str, event_id: str, mode: str) -> Optional[Dict[str, Any]]:
    """
    Load analysis document from S3.

    Returns:
        Dict with 'content', 'timestamp', 'mode' or None if not found
    """
    try:
        s3_client = get_s3_client()
        key = get_s3_analysis_doc_key(agent_name, event_id, mode)

        response = s3_client.get_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))

        logger.info(f"Loaded {mode} analysis from S3: {key}")
        return data
    except s3_client.exceptions.NoSuchKey:
        logger.info(f"No S3 analysis document found for {agent_name}/{event_id}/{mode}")
        return None
    except Exception as e:
        logger.error(f"Error loading analysis from S3: {e}", exc_info=True)
        return None


def save_analysis_doc_to_s3(agent_name: str, event_id: str, mode: str, content: str) -> bool:
    """
    Save analysis document to S3.

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
        key = get_s3_analysis_doc_key(agent_name, event_id, mode)

        data = {
            'content': content,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'mode': mode,
            'agent': agent_name,
            'event': event_id
        }

        s3_client.put_object(
            Bucket=CANVAS_ANALYSIS_BUCKET,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )

        logger.info(f"Saved {mode} analysis to S3: {key}")
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
    groups_read_mode: str = 'none',
    event_type: str = 'shared',
    personal_layer: Optional[str] = None,
    personal_event_id: Optional[str] = None
) -> Optional[str]:
    """
    Get analysis document for the specified mode, generating if needed.
    Uses hybrid storage: Memory cache → S3 → Generate new

    Args:
        agent_name: Agent name
        event_id: Event ID
        depth_mode: One of 'mirror', 'lens', 'portal'
        force_refresh: If True, bypass cache and regenerate
        groups_read_mode: Groups read mode from settings
        event_type: Event type
        personal_layer: Personal agent layer
        personal_event_id: Personal event ID

    Returns:
        Analysis document markdown string or None
    """
    cache_key = f"{agent_name}_{event_id}_{depth_mode}"

    # 1. Check memory cache (unless forced refresh)
    if not force_refresh and cache_key in CANVAS_ANALYSIS_CACHE:
        cached = CANVAS_ANALYSIS_CACHE[cache_key]
        age_seconds = (datetime.now(timezone.utc) - cached['timestamp']).total_seconds()
        age_minutes = age_seconds / 60

        if age_minutes < ANALYSIS_CACHE_TTL_MINUTES:
            logger.info(f"Using memory cached {depth_mode} analysis for {agent_name}/{event_id} (age: {age_minutes:.1f}m)")
            return cached['doc']
        else:
            logger.info(f"Memory cached {depth_mode} analysis expired (age: {age_minutes:.1f}m)")

    # 2. Check S3 storage (unless forced refresh)
    if not force_refresh:
        s3_doc_data = load_analysis_doc_from_s3(agent_name, event_id, depth_mode)
        if s3_doc_data:
            # Check if S3 doc is fresh enough (within TTL)
            try:
                s3_timestamp = datetime.fromisoformat(s3_doc_data['timestamp'])
                age_seconds = (datetime.now(timezone.utc) - s3_timestamp).total_seconds()
                age_minutes = age_seconds / 60

                if age_minutes < ANALYSIS_CACHE_TTL_MINUTES:
                    logger.info(f"Using S3 cached {depth_mode} analysis for {agent_name}/{event_id} (age: {age_minutes:.1f}m)")

                    # Restore to memory cache
                    CANVAS_ANALYSIS_CACHE[cache_key] = {
                        'doc': s3_doc_data['content'],
                        'timestamp': s3_timestamp,
                        'mode': depth_mode,
                        'agent': agent_name,
                        'event': event_id
                    }

                    return s3_doc_data['content']
                else:
                    logger.info(f"S3 cached {depth_mode} analysis expired (age: {age_minutes:.1f}m)")
            except Exception as e:
                logger.warning(f"Error parsing S3 timestamp: {e}")

    # 3. Generate new analysis
    logger.info(f"Generating fresh {depth_mode} analysis for {agent_name}/{event_id}")

    # Get transcript content based on Settings > Memory
    transcript_content = get_transcript_content_for_analysis(agent_name, event_id, groups_read_mode)

    if not transcript_content:
        logger.warning(f"No transcript content available for analysis")
        return None

    # Run analysis agent
    analysis_doc = run_analysis_agent(
        agent_name=agent_name,
        event_id=event_id,
        mode=depth_mode,
        transcript_content=transcript_content,
        event_type=event_type,
        personal_layer=personal_layer,
        personal_event_id=personal_event_id
    )

    if not analysis_doc:
        logger.error(f"Failed to generate {depth_mode} analysis")
        return None

    # 4. Save to S3
    save_analysis_doc_to_s3(agent_name, event_id, depth_mode, analysis_doc)

    # 5. Cache result in memory
    CANVAS_ANALYSIS_CACHE[cache_key] = {
        'doc': analysis_doc,
        'timestamp': datetime.now(timezone.utc),
        'mode': depth_mode,
        'agent': agent_name,
        'event': event_id
    }

    logger.info(f"Cached {depth_mode} analysis in memory and S3 for {agent_name}/{event_id}")
    return analysis_doc


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
    s3_doc_data = load_analysis_doc_from_s3(agent_name, event_id, depth_mode)
    if s3_doc_data:
        try:
            s3_timestamp = datetime.fromisoformat(s3_doc_data['timestamp'])
            age_seconds = (datetime.now(timezone.utc) - s3_timestamp).total_seconds()
            age_minutes = age_seconds / 60

            if age_minutes < ANALYSIS_CACHE_TTL_MINUTES:
                return {
                    'state': 'ready',
                    'timestamp': s3_timestamp.isoformat()
                }
        except Exception as e:
            logger.warning(f"Error parsing S3 timestamp for status: {e}")

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
