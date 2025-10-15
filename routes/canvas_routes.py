"""
Canvas view streaming routes.
Handles PTT-to-LLM streaming for the canvas interface.
"""

import os
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from flask import jsonify, Response, stream_with_context, g
from gotrue.types import User as SupabaseUser
from anthropic import Anthropic, AnthropicError
from tenacity import RetryError
from utils.s3_utils import get_cached_s3_file, find_file_any_extension, get_objective_function
from utils.api_key_manager import get_api_key
from utils.canvas_analysis_agents import get_or_generate_analysis_doc, get_analysis_status

logger = logging.getLogger(__name__)


def get_agent_specific_prompt_only(agent_name: str) -> str:
    """
    Load ONLY the agent-specific prompt without any base system prompt.
    This ensures canvas doesn't inherit the full taxonomy structure.
    """
    agent_pattern = f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}'
    agent_prompt = get_cached_s3_file(
        cache_key=agent_pattern,
        description=f"agent system prompt for {agent_name}",
        fetch_function=lambda: find_file_any_extension(agent_pattern, f"agent system prompt for {agent_name}")
    )
    if agent_prompt:
        logger.info(f"Loaded agent-specific canvas prompt for '{agent_name}' ({len(agent_prompt)} chars)")
        return agent_prompt
    else:
        logger.warning(f"No agent-specific prompt found for '{agent_name}', using empty string")
        return ""


def get_canvas_base_and_depth_prompt(depth_mode: str) -> str:
    """
    Returns canvas base prompt combined with MLP depth-specific instructions.

    Structure:
    1. Canvas Base (global rules for all depths)
    2. MLP Depth Instructions (mirror/lens/portal specific)

    Args:
        depth_mode: One of 'mirror', 'lens', or 'portal'

    Returns:
        Combined canvas base + depth instructions
    """
    # Canvas Base: Universal rules for all canvas interactions
    canvas_base = """=== CANVAS BASE ===
You are a specialized canvas agent - a thoughtful, warm advisor responding on a visual interface using voice input.

STRICT BREVITY REQUIREMENTS:
- Brevity: Maximum 2 short sentences per response (1 sentence strongly preferred)
- Be EXTREMELY succinct - every word must count
- Mirror back insights directly without preamble or setup
- Plain text only - NO markdown formatting (no **, -, #, etc.)
- NO lists, bullet points, or special characters
- NO meta-commentary like "Looking at...", "Based on...", "The analysis shows..."
- NO process language like "Examining...", "Considering...", "Reviewing..."
- NO quoting or using phrases like "As X said..." or "One person noted..."
- NO example patterns or framing phrases
- Just state the insight directly as if it's your own observation

RESPONSE STYLE & VOICE:
- Speak directly to the user as their warm yet effective advisor
- Use "I" when referring to yourself - own your insights personally
- Use "you" when addressing the user/audience - NEVER use "they" when referring to people in the conversation
- Present insights as clear, confident statements
- Be conversational and human - brief doesn't mean cold
- Each response is a single focused thought with warmth
- Be on-point and actionable while staying approachable

WHEN NO TRANSCRIPT/ANALYSIS IS AVAILABLE:
- Keep the conversation going naturally
- Respond to the user's question directly based on what they asked
- Don't apologize or mention lack of context
- Use your general knowledge and understanding to provide value
- Stay warm, helpful, and engaged

KNOWLEDGE SOURCES:
- You draw on OBJECTIVE FUNCTION, AGENT CONTEXT, RAW SOURCES, RETRIEVED MEMORY, and ANALYSIS DOCUMENTS (when provided)
- RAW SOURCES contain unprocessed transcripts and documents for verification and specific quotes
- ANALYSIS DOCUMENTS provide interpreted insights (current and previous)
- CURRENT ANALYSIS is your primary analysis knowledge source
- PREVIOUS ANALYSIS provides historical analysis context
- Synthesize these naturally - never mention them explicitly
- If no analysis exists, respond based on the user's message using your general understanding

SOURCE DIFFERENTIATION:
- Analysis documents may contain insights from MULTIPLE SOURCES (e.g., different breakout groups, transcript files, or events)
- When responding, preserve source differentiation when it genuinely matters to the user's question
- If asked about "breakout 1" specifically, draw only from that source's insights
- If asked about differences between groups, contrast them naturally
- Use "you" when addressing the group: "You focused on..." not "They focused on..."
- If the pattern is universal across sources, state it confidently without qualifying
- Blend sources thoughtfully - differentiate when it matters, synthesize when it doesn't

WISDOM IN PRACTICE:
- Trust your audience's intelligence - they can handle nuance
- Sometimes the most helpful response is also the simplest one
- Stay curious and humble about what you don't know

=== END CANVAS BASE ==="""

    # MLP Framework: Mirror/Lens/Portal depth instructions
    mlp_instructions = {
        'mirror': """
=== MIRROR MODE (Edges) ===
Surface explicit but peripheral information - the edge cases that were stated but sit at the margins.

Focus on:
- Minority viewpoints voiced in the conversation
- Side comments and tangential observations
- Outlier perspectives that were mentioned
- Peripheral but concrete concerns

State these edge cases directly. No interpretation, no attribution phrases, no quoting.
Simply present the peripheral insight as a clear observation.
=== END MIRROR MODE ===
""",

        'lens': """
=== LENS MODE (Latent Needs) ===
Identify hidden patterns and unspoken requirements implied by the conversation.

Focus on:
- Recurring themes across different speakers
- Emotional undercurrents and group dynamics
- Systemic issues beneath surface symptoms
- Paradoxes and contradictions
- What's being avoided or protected

State the deeper need directly. No hedging phrases, no analytical preambles.
Present the latent pattern as a confident insight.
=== END LENS MODE ===
""",

        'portal': """
=== PORTAL MODE (Questions) ===
Ask transformative questions that open possibility spaces, derived from deeper patterns.

Focus on:
- Questions challenging limiting assumptions
- Questions revealing transformation opportunities
- Questions about paradigm shifts
- Questions predicting intervention outcomes

Ask the question directly. No setup, no explanation of where it comes from.
Just pose the possibility clearly and powerfully.
=== END PORTAL MODE ===
"""
    }

    # Combine canvas base with selected depth mode
    depth_instructions = mlp_instructions.get(depth_mode, mlp_instructions['mirror'])
    return f"{canvas_base}\n\n{depth_instructions}"


def register_canvas_routes(app, anthropic_client, supabase_auth_required):
    """
    Register canvas-related routes to the Flask app.

    Args:
        app: Flask application instance
        anthropic_client: Initialized Anthropic client
        supabase_auth_required: Auth decorator function
    """

    @app.route('/api/canvas/stream', methods=['POST'])
    @supabase_auth_required(agent_required=True)
    def handle_canvas_stream(user: SupabaseUser):
        """
        Handles canvas view streaming requests with Claude 4.5 Sonnet.
        Designed for concise, visually-oriented responses optimized for canvas display.
        """
        logger.info(f"Received POST request to /api/canvas/stream from user: {user.id}")

        try:
            data = g.get('json_data', {})
            if not data or 'transcript' not in data:
                return jsonify({"error": "Missing 'transcript' in request body"}), 400
        except Exception as e:
            logger.error(f"Error accessing request data: {e}")
            return jsonify({"error": "Invalid request data"}), 400

        # Extract request parameters
        agent_name = data.get('agent')
        transcript_text = data.get('transcript', '')
        depth_mode = data.get('depth', 'mirror')  # mirror | lens | portal
        conversation_history = data.get('history', [])  # Array of {role, content} messages
        client_timezone = data.get('timezone', 'UTC')  # Client timezone
        force_refresh_analysis = data.get('forceRefreshAnalysis', False)  # Manual refresh button
        clear_previous_analysis = data.get('clearPrevious', False)  # Clear previous on new context
        individual_raw_transcript_toggle_states = data.get('individualRawTranscriptToggleStates', {})  # For "some" mode
        event_id = '0000'  # Canvas always uses event 0000
        model_selection = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-5-20250929")
        temperature = 0.7

        # Get transcript_listen_mode and groups_read_mode from request (consistent with main chat)
        # FIXED: Read from request payload instead of Supabase (column doesn't exist)
        transcript_listen_mode = data.get('transcriptListenMode', 'latest')
        groups_read_mode = data.get('groupsReadMode', 'none')
        logger.info(f"Canvas: transcript_listen_mode={transcript_listen_mode}, groups_read_mode={groups_read_mode} for {agent_name}")

        # Get event access profile (same as main chat agent - api_server.py:5208)
        from api_server import get_event_access_profile
        event_profile = get_event_access_profile(agent_name, user.id)
        allowed_events = set(event_profile.get('allowed_events') or {"0000"}) if event_profile else {"0000"}
        event_types_map = dict(event_profile.get('event_types') or {}) if event_profile else {}
        logger.info(f"Canvas: event_profile loaded with {len(allowed_events)} allowed events, {len(event_types_map)} event types")

        # Get per-agent custom API key or fallback to default
        agent_anthropic_key = get_api_key(agent_name, 'anthropic')
        if not agent_anthropic_key:
            logger.error(f"Canvas stream fail: No Anthropic API key available for agent '{agent_name}'.")
            return jsonify({"error": "AI service unavailable"}), 503

        # Create agent-specific Anthropic client
        try:
            agent_anthropic_client = Anthropic(api_key=agent_anthropic_key)
            logger.info(f"Anthropic client initialized for agent '{agent_name}' (custom key: {agent_anthropic_key != os.getenv('ANTHROPIC_API_KEY')})")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client for agent '{agent_name}': {e}", exc_info=True)
            return jsonify({"error": "AI service initialization failed"}), 503

        def generate_canvas_stream():
            """Generator for canvas-specific streaming responses."""
            try:
                logger.info(f"Canvas stream started for Agent: {agent_name}, User: {user.id}, Depth: {depth_mode}, Force refresh: {force_refresh_analysis}, Clear previous: {clear_previous_analysis}")

                # OPTION C (HYBRID): Load ALL three analysis documents (mirror, lens, portal)
                # The depth_mode becomes an "emphasis hint" rather than a filter
                analyses = {}
                try:
                    for mode in ['mirror', 'lens', 'portal']:
                        current_doc, previous_doc = get_or_generate_analysis_doc(
                            agent_name=agent_name,
                            event_id=event_id,
                            depth_mode=mode,
                            force_refresh=force_refresh_analysis,
                            clear_previous=clear_previous_analysis,
                            transcript_listen_mode=transcript_listen_mode,
                            groups_read_mode=groups_read_mode,
                            individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states,
                            event_type='shared',  # Canvas typically uses shared context
                            personal_layer=None,  # Canvas doesn't use personal layers for now
                            personal_event_id=None,
                            allowed_events=allowed_events,
                            event_types_map=event_types_map,
                            event_profile=event_profile
                        )

                        analyses[mode] = {
                            'current': current_doc,
                            'previous': previous_doc
                        }

                        if current_doc:
                            logger.info(f"Canvas: Loaded current {mode} analysis ({len(current_doc)} chars)")
                        if previous_doc:
                            logger.info(f"Canvas: Loaded previous {mode} analysis ({len(previous_doc)} chars)")
                        if not current_doc:
                            logger.info(f"Canvas: No current {mode} analysis available")

                except Exception as analysis_err:
                    logger.error(f"Error getting analysis documents: {analysis_err}", exc_info=True)
                    # Initialize empty analyses if error occurs
                    for mode in ['mirror', 'lens', 'portal']:
                        if mode not in analyses:
                            analyses[mode] = {'current': None, 'previous': None}

                # Get raw transcript content based on Settings toggles
                raw_transcript_content = None
                try:
                    from utils.canvas_analysis_agents import get_all_canvas_source_content

                    raw_transcript_content = get_all_canvas_source_content(
                        agent_name=agent_name,
                        event_id=event_id,
                        transcript_listen_mode=transcript_listen_mode,
                        groups_read_mode=groups_read_mode,
                        individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states,
                        allowed_events=allowed_events,
                        event_types_map=event_types_map,
                        event_profile=event_profile
                    )

                    if raw_transcript_content:
                        logger.info(f"Canvas: Loaded raw source content ({len(raw_transcript_content)} chars)")
                    else:
                        logger.info("Canvas: No raw source content available")
                except Exception as source_err:
                    logger.error(f"Error loading source content for canvas: {source_err}", exc_info=True)
                    raw_transcript_content = None

                # RAG retrieval for canvas agent
                rag_context = None
                rag_docs = None
                try:
                    from utils.retrieval_handler import RetrievalHandler

                    retriever = RetrievalHandler(
                        index_name="river",
                        agent_name=agent_name,
                        event_id=event_id,
                        anthropic_api_key=agent_anthropic_key,
                        openai_api_key=get_api_key(agent_name, 'openai'),
                        event_type='shared',
                        include_t3=True,
                    )

                    # Query based on user message
                    rag_docs = retriever.get_relevant_context_tiered(
                        query=transcript_text,
                        tier_caps=[4, 8, 6, 6, 4],
                        include_t3=True
                    )

                    if rag_docs:
                        rag_parts = []
                        for doc in rag_docs:
                            source_label = doc.metadata.get('source_label', '')
                            filename = doc.metadata.get('filename', 'unknown')
                            tier = doc.metadata.get('retrieval_tier', 'unknown')
                            score = doc.metadata.get('score', 0)
                            age_display = doc.metadata.get('age_display', 'unknown age')

                            rag_parts.append(
                                f"[{tier}] {filename} (score: {score:.3f}, age: {age_display})\n{doc.page_content}"
                            )

                        rag_context = "\n\n---\n\n".join(rag_parts)
                        logger.info(f"Canvas: Retrieved {len(rag_docs)} context docs")
                    else:
                        logger.info("Canvas: No RAG context retrieved")

                except Exception as rag_err:
                    logger.error(f"Canvas RAG retrieval error: {rag_err}", exc_info=True)

                # Build canvas system prompt (lightweight, NO full taxonomy)
                # Structure: Canvas Base + MLP Depth + Objective Function + Analysis Doc (if available) + Agent-Specific + Time
                canvas_base_and_depth = get_canvas_base_and_depth_prompt(depth_mode)
                agent_specific = get_agent_specific_prompt_only(agent_name)

                # Load objective function (agent-specific or global fallback)
                objective_function = None
                try:
                    objective_function = get_objective_function(agent_name) or get_objective_function(None)
                    if objective_function:
                        logger.info(f"Canvas: Loaded objective function for '{agent_name}' ({len(objective_function)} chars)")
                    else:
                        logger.info(f"Canvas: No objective function found for '{agent_name}'")
                except Exception as obj_err:
                    logger.warning(f"Error loading objective function for canvas: {obj_err}")
                    objective_function = None

                # Add current time in user's timezone
                try:
                    user_tz = ZoneInfo(client_timezone)
                    current_user_time = datetime.now(user_tz)
                    current_user_time_str = current_user_time.strftime('%Y-%m-%d %H:%M:%S %Z')
                    tz_abbr = current_user_time.strftime('%Z')
                except Exception as e:
                    logger.warning(f"Invalid timezone '{client_timezone}', falling back to UTC: {e}")
                    current_user_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    tz_abbr = 'UTC'

                time_section = f"\n\n=== CURRENT TIME ===\nCurrent date and time: {current_user_time_str} (user's local timezone: {client_timezone})\n=== END CURRENT TIME ==="

                # Combine: Objective → Agent → Canvas Base + Depth → Previous Analyses → Current Analyses → Mode Emphasis → Time
                # (Tree structure: roots → stem → trunk → branches → branches → leaves)
                system_prompt = ""

                # 1. Objective function (roots - why you exist)
                if objective_function:
                    system_prompt += f"=== OBJECTIVE FUNCTION ===\n{objective_function}\n=== END OBJECTIVE FUNCTION ==="

                # 2. Agent context (stem - who you are)
                if agent_specific:
                    system_prompt += f"\n\n=== AGENT CONTEXT ===\n{agent_specific}\n=== END AGENT CONTEXT ==="

                # 3. RAG retrieved memory (contextual knowledge)
                if rag_context:
                    system_prompt += f"\n\n=== RETRIEVED MEMORY ===\n"
                    system_prompt += "The following context was retrieved from long-term memory based on your current query.\n"
                    system_prompt += "Draw on this when relevant to provide informed, contextual responses.\n\n"
                    system_prompt += rag_context
                    system_prompt += f"\n=== END RETRIEVED MEMORY ==="

                # 4. Canvas base + MLP depth (trunk - how you operate)
                system_prompt += f"\n\n{canvas_base_and_depth}"

                # 5. Raw sources (unprocessed content - transcripts and documents)
                if raw_transcript_content:
                    system_prompt += f"\n\n=== RAW SOURCES ===\n"
                    system_prompt += "You have access to the raw source content that was used to generate the analyses below.\n"
                    system_prompt += "Use this to verify analysis insights, find specific quotes, or surface details not captured in the analysis.\n"
                    system_prompt += "When answering questions about specific statements or exact wording, reference these raw sources.\n\n"
                    system_prompt += raw_transcript_content
                    system_prompt += "\n=== END RAW SOURCES ==="

                # 6. Previous analyses (branches - historical context for all modes)
                has_previous_analyses = any(analyses[mode]['previous'] for mode in ['mirror', 'lens', 'portal'])
                if has_previous_analyses:
                    system_prompt += f"\n\n=== PREVIOUS ANALYSES (HISTORICAL CONTEXT) ==="
                    for mode in ['mirror', 'lens', 'portal']:
                        if analyses[mode]['previous']:
                            mode_label = mode.upper()
                            system_prompt += f"\n\n--- Previous {mode_label} Analysis ---\n{analyses[mode]['previous']}"
                    system_prompt += f"\n\n=== END PREVIOUS ANALYSES ==="

                # 7. Current analyses (branches - fresh content for all modes)
                has_current_analyses = any(analyses[mode]['current'] for mode in ['mirror', 'lens', 'portal'])
                if has_current_analyses:
                    system_prompt += f"\n\n=== CURRENT ANALYSES (COMPREHENSIVE CONTEXT) ==="
                    system_prompt += f"\nYou have access to three complementary analyses of the same conversation:\n"

                    for mode in ['mirror', 'lens', 'portal']:
                        if analyses[mode]['current']:
                            mode_label = mode.upper()
                            mode_desc = {
                                'mirror': 'Explicit information (center and edge cases)',
                                'lens': 'Hidden patterns and latent needs',
                                'portal': 'Transformative questions and possibilities'
                            }[mode]
                            system_prompt += f"\n--- {mode_label} Analysis: {mode_desc} ---\n{analyses[mode]['current']}\n"

                    system_prompt += f"=== END CURRENT ANALYSES ==="

                # 8. Mode emphasis (guidance based on selected mode)
                if depth_mode == 'mirror':
                    mode_emphasis_text = "\n=== MODE EMPHASIS: MIRROR ===\nThe user has selected MIRROR mode. When relevant to their question:\n- Prioritize insights from the Mirror analysis (explicit/peripheral information)\n- Surface edge cases and minority viewpoints\n- Focus on what was actually stated but sits at the margins\nHowever, you may draw on Lens or Portal analyses if they better serve the user's question.\n=== END MODE EMPHASIS ==="
                elif depth_mode == 'lens':
                    mode_emphasis_text = "\n=== MODE EMPHASIS: LENS ===\nThe user has selected LENS mode. When relevant to their question:\n- Prioritize insights from the Lens analysis (hidden patterns/latent needs)\n- Surface recurring themes and systemic issues\n- Focus on what's implied but not explicitly stated\nHowever, you may draw on Mirror or Portal analyses if they better serve the user's question.\n=== END MODE EMPHASIS ==="
                elif depth_mode == 'portal':
                    mode_emphasis_text = "\n=== MODE EMPHASIS: PORTAL ===\nThe user has selected PORTAL mode. When relevant to their question:\n- Prioritize insights from the Portal analysis (transformative questions)\n- Frame responses as possibilities and interventions\n- Focus on opening new possibility spaces\nHowever, you may draw on Mirror or Lens analyses if they better serve the user's question.\n=== END MODE EMPHASIS ==="
                else:
                    mode_emphasis_text = "\n=== MODE EMPHASIS: MIRROR ===\nThe user has selected MIRROR mode. When relevant to their question:\n- Prioritize insights from the Mirror analysis (explicit/peripheral information)\n- Surface edge cases and minority viewpoints\n- Focus on what was actually stated but sits at the margins\nHowever, you may draw on Lens or Portal analyses if they better serve the user's question.\n=== END MODE EMPHASIS ==="

                system_prompt += f"\n\n{mode_emphasis_text}"

                # 9. CRITICAL: Brevity booster (reinforce after heavy context)
                brevity_booster = """

=== CRITICAL REMINDER ===
Your responses must be EXTREMELY brief:
- Maximum 2 sentences (1 sentence preferred)
- NO markdown formatting whatsoever (no **, -, #, bullets)
- NO meta-commentary or preambles
- State insights directly and confidently
This is a voice interface - every word must count.
=== END REMINDER ==="""
                system_prompt += brevity_booster

                # 10. Current time (leaves - immediate moment)
                system_prompt += time_section

                # Build prompt type description for logging (matches tree order)
                prompt_components = []
                if objective_function:
                    prompt_components.append("objective")
                if agent_specific:
                    prompt_components.append("agent")

                # Track RAG context if present
                if rag_context:
                    prompt_components.append(f"rag({len(rag_docs)}docs)")

                prompt_components.extend(["base", "depth"])

                # Track raw sources if present
                if raw_transcript_content:
                    prompt_components.append(f"raw_sources({len(raw_transcript_content)})")

                # Count loaded analyses (all three modes)
                loaded_current = [mode for mode in ['mirror', 'lens', 'portal'] if analyses[mode]['current']]
                loaded_previous = [mode for mode in ['mirror', 'lens', 'portal'] if analyses[mode]['previous']]

                analysis_status = []
                if loaded_previous:
                    analysis_status.append(f"previous({len(loaded_previous)})")
                if loaded_current:
                    analysis_status.append(f"current({len(loaded_current)})")

                if analysis_status:
                    prompt_components.append(f"analyses({'+'.join(analysis_status)})")
                else:
                    prompt_components.append("analyses(none)")

                prompt_components.append(f"emphasis({depth_mode})")
                prompt_components.append("time")
                prompt_type = "+".join(prompt_components)
                logger.info(f"Canvas system prompt built (HYBRID): {len(system_prompt)} chars ({prompt_type})")

                # Build messages with conversation history
                messages = []
                if conversation_history:
                    messages.extend(conversation_history)
                    logger.info(f"Canvas: Using conversation history with {len(conversation_history)} messages")
                else:
                    logger.info("Canvas: No conversation history provided")

                # Add current user message
                messages.append({"role": "user", "content": transcript_text})

                logger.info(f"Calling Anthropic API for canvas stream (model: {model_selection}, depth: {depth_mode}, total messages: {len(messages)})")

                # Stream from Anthropic using agent-specific client
                with agent_anthropic_client.messages.stream(
                    model=model_selection,
                    max_tokens=512,  # Reduced from 4096 to enforce brevity
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages
                ) as stream:
                    for chunk in stream.text_stream:
                        if chunk:
                            yield f"data: {json.dumps({'delta': chunk})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

                # Reinforce retrieved memories
                if rag_docs:
                    try:
                        retriever.reinforce_memories(rag_docs)
                        logger.info(f"Canvas: Reinforced {len(rag_docs)} memories")
                    except Exception as reinforce_err:
                        logger.error(f"Canvas memory reinforcement error: {reinforce_err}")

                logger.info(f"Canvas stream completed successfully for agent {agent_name}")

            except (AnthropicError, RetryError) as e:
                logger.error(f"Anthropic API error in canvas stream: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': f'AI service error: {str(e)}'})}\n\n"
            except Exception as e:
                logger.error(f"Error in canvas generate_stream: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': 'An internal server error occurred during the stream.'})}\n\n"

        return Response(stream_with_context(generate_canvas_stream()), mimetype='text/event-stream')

    @app.route('/api/canvas/analysis/status', methods=['GET'])
    @supabase_auth_required(agent_required=True)
    def get_canvas_analysis_status(user: SupabaseUser):
        """
        Get analysis status for the canvas interface.
        Returns status for the currently selected mode.
        """
        try:
            agent_name = g.get('agent_name')
            depth_mode = g.get('json_data', {}).get('depth', 'mirror')
            event_id = '0000'

            if not agent_name:
                return jsonify({"error": "Missing agent parameter"}), 400

            status = get_analysis_status(agent_name, event_id, depth_mode)
            return jsonify(status)

        except Exception as e:
            logger.error(f"Error getting canvas analysis status: {e}", exc_info=True)
            return jsonify({"error": "Failed to get analysis status"}), 500

    @app.route('/api/canvas/analysis/refresh', methods=['POST'])
    @supabase_auth_required(agent_required=True)
    def refresh_canvas_analysis(user: SupabaseUser):
        """
        Manually trigger refresh of all analysis documents (mirror, lens, portal).
        This is called when the user clicks the sparkles icon.
        """
        logger.info(f"Received POST request to /api/canvas/analysis/refresh from user: {user.id}")

        try:
            data = g.get('json_data', {})
            agent_name = data.get('agent')
            clear_previous = data.get('clearPrevious', False)  # Clear previous on new context
            individual_raw_transcript_toggle_states = data.get('individualRawTranscriptToggleStates', {})  # For "some" mode
            event_id = '0000'

            if not agent_name:
                return jsonify({"error": "Missing agent parameter"}), 400

            # Get transcript_listen_mode and groups_read_mode from request (consistent with main chat)
            # FIXED: Read from request payload instead of Supabase (column doesn't exist)
            transcript_listen_mode = data.get('transcriptListenMode', 'latest')
            groups_read_mode = data.get('groupsReadMode', 'none')
            logger.info(f"Canvas refresh: transcript_listen_mode={transcript_listen_mode}, groups_read_mode={groups_read_mode} for {agent_name}")

            # Get event access profile (same as main chat agent - api_server.py:5208)
            from api_server import get_event_access_profile
            event_profile = get_event_access_profile(agent_name, user.id)
            allowed_events = set(event_profile.get('allowed_events') or {"0000"}) if event_profile else {"0000"}
            event_types_map = dict(event_profile.get('event_types') or {}) if event_profile else {}

            # Refresh all three modes in parallel (for now, sequential is simpler)
            results = {}
            for mode in ['mirror', 'lens', 'portal']:
                try:
                    logger.info(f"Refreshing {mode} analysis for {agent_name} (clear_previous={clear_previous})")
                    doc, _ = get_or_generate_analysis_doc(
                        agent_name=agent_name,
                        event_id=event_id,
                        depth_mode=mode,
                        force_refresh=True,  # Force refresh
                        clear_previous=clear_previous,  # Clear previous if requested
                        transcript_listen_mode=transcript_listen_mode,
                        groups_read_mode=groups_read_mode,
                        individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states,
                        event_type='shared',
                        personal_layer=None,
                        personal_event_id=None,
                        allowed_events=allowed_events,
                        event_types_map=event_types_map,
                        event_profile=event_profile
                    )
                    results[mode] = {
                        'success': doc is not None,
                        'length': len(doc) if doc else 0
                    }
                    if doc:
                        logger.info(f"Successfully refreshed {mode} analysis ({len(doc)} chars)")
                    else:
                        logger.warning(f"Failed to refresh {mode} analysis")
                except Exception as e:
                    logger.error(f"Error refreshing {mode} analysis: {e}", exc_info=True)
                    results[mode] = {'success': False, 'error': str(e)}

            # Get status for response
            status = get_analysis_status(agent_name, event_id, 'mirror')  # Can use any mode

            return jsonify({
                'success': True,
                'results': results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error refreshing canvas analysis: {e}", exc_info=True)
            return jsonify({"error": "Failed to refresh analysis"}), 500
