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
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document identifying HIDDEN INFORMATION at both surface and depth levels.

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
Analyze the transcript(s) thoroughly and produce a COMPREHENSIVE markdown document composed ENTIRELY OF QUESTIONS that open possibility spaces.

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
