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
from utils.s3_utils import get_latest_system_prompt
from utils.api_key_manager import get_api_key

logger = logging.getLogger(__name__)


def register_canvas_routes(app, anthropic_client, supabase_auth_required):
    """
    Register canvas-related routes to the Flask app.

    Args:
        app: Flask application instance
        anthropic_client: Initialized Anthropic client
        supabase_auth_required: Auth decorator function
    """

    @app.route('/api/canvas/warmup', methods=['POST'])
    @supabase_auth_required(agent_required=True)
    def handle_canvas_warmup(user: SupabaseUser):
        """
        Warmup endpoint to prime Anthropic Claude prompt caches.
        Prevents cold starts on first canvas interaction.

        Note: OpenAI Whisper warmup would require sending audio files,
        so only Anthropic is warmed up here. The 5-8s TTFT on first message
        is reduced to 1-2s on subsequent messages after prompt caching.
        """
        logger.info(f"Canvas warmup request from user: {user.id}")

        try:
            data = g.get('json_data', {})
            agent_name = data.get('agent')

            if not agent_name:
                return jsonify({"error": "Missing 'agent' in request body"}), 400

            # Get per-agent custom API key or fallback to default
            agent_anthropic_key = get_api_key(agent_name, 'anthropic')
            if not agent_anthropic_key:
                logger.warning(f"Canvas warmup: No Anthropic API key available for agent '{agent_name}'.")
                return jsonify({"status": "skipped", "reason": "No API key"}), 200

            # Create agent-specific Anthropic client
            try:
                agent_anthropic_client = Anthropic(api_key=agent_anthropic_key)
            except Exception as e:
                logger.error(f"Canvas warmup: Failed to initialize Anthropic client: {e}")
                return jsonify({"status": "skipped", "reason": "Client init failed"}), 200

            # Load agent system prompt to cache it
            agent_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
            depth_instructions = get_canvas_depth_instructions('mirror')

            # Minimal warmup prompt
            try:
                user_tz = ZoneInfo('UTC')
                current_time = datetime.now(user_tz).strftime('%Y-%m-%d %H:%M:%S UTC')
            except Exception:
                current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

            time_section = f"\n\n=== CURRENT TIME ===\nYour internal clock shows the current date and time is: **{current_time}** (user's local timezone: UTC).\n=== END CURRENT TIME ==="
            system_prompt = f"{agent_system_prompt}\n\n{depth_instructions}{time_section}"

            # Send minimal message to prime cache (non-streaming for speed)
            model_selection = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-5-20250929")

            logger.info(f"Canvas warmup: Sending cache-priming request for agent '{agent_name}'")
            response = agent_anthropic_client.messages.create(
                model=model_selection,
                max_tokens=10,  # Minimal tokens
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": "Hi"}]
            )

            logger.info(f"Canvas warmup: Successfully primed cache for agent '{agent_name}'")
            return jsonify({"status": "success", "agent": agent_name}), 200

        except Exception as e:
            logger.error(f"Canvas warmup error: {e}", exc_info=True)
            # Don't fail hard - warmup is optional
            return jsonify({"status": "error", "message": str(e)}), 200

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
        model_selection = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-5-20250929")
        temperature = 0.7

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
                logger.info(f"Canvas stream started for Agent: {agent_name}, User: {user.id}, Depth: {depth_mode}")

                # Load agent-specific system prompt from S3
                agent_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
                logger.info(f"Loaded agent system prompt for '{agent_name}' ({len(agent_system_prompt)} chars)")

                # Combine agent prompt with canvas-specific depth instructions
                depth_instructions = get_canvas_depth_instructions(depth_mode)

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

                time_section = f"\n\n=== CURRENT TIME ===\nYour internal clock shows the current date and time is: **{current_user_time_str}** (user's local timezone: {client_timezone}).\n=== END CURRENT TIME ==="

                system_prompt = f"{agent_system_prompt}\n\n{depth_instructions}{time_section}"

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
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages
                ) as stream:
                    for chunk in stream.text_stream:
                        if chunk:
                            yield f"data: {json.dumps({'delta': chunk})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                logger.info(f"Canvas stream completed successfully for agent {agent_name}")

            except (AnthropicError, RetryError) as e:
                logger.error(f"Anthropic API error in canvas stream: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': f'AI service error: {str(e)}'})}\n\n"
            except Exception as e:
                logger.error(f"Error in canvas generate_stream: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': 'An internal server error occurred during the stream.'})}\n\n"

        return Response(stream_with_context(generate_canvas_stream()), mimetype='text/event-stream')


def get_canvas_depth_instructions(depth_mode: str) -> str:
    """
    Returns canvas-specific depth instructions to append to agent system prompt.

    Args:
        depth_mode: One of 'mirror', 'lens', or 'portal'

    Returns:
        Depth-specific instructions for response style
    """
    depth_instructions = {
        'mirror': """=== CANVAS MODE: MIRROR ===
You are now responding in Canvas mode with MIRROR depth.

CRITICAL FORMATTING RULES:
- NO markdown allowed (no **, -, #, etc.)
- Plain text only
- Keep responses SHORT (2-4 sentences max)
- Responses should fit on screen without scrolling

RESPONSE STYLE:
- Concise, visually-impactful responses
- Direct and insightful
- Clear, powerful language for large display
- Strong, declarative statements
- No unnecessary preamble

You are responding to voice input on a visual canvas.""",

        'lens': """=== CANVAS MODE: LENS ===
You are now responding in Canvas mode with LENS depth.

CRITICAL FORMATTING RULES:
- NO markdown allowed (no **, -, #, etc.)
- Plain text only
- Keep responses MODERATE (4-6 sentences max)
- Responses should fit on screen without scrolling

RESPONSE STYLE:
- Structured, insightful analysis
- Break down ideas systematically
- Clear sections using line breaks (not markdown)
- Deeper analysis while remaining concise
- Highlight connections

You are responding to voice input on a visual canvas.""",

        'portal': """=== CANVAS MODE: PORTAL ===
You are now responding in Canvas mode with PORTAL depth.

CRITICAL FORMATTING RULES:
- NO markdown allowed (no **, -, #, etc.)
- Plain text only
- Keep responses COMPREHENSIVE but BRIEF (6-10 sentences max)
- Responses should fit on screen without scrolling

RESPONSE STYLE:
- Comprehensive, multifaceted responses
- Multiple perspectives
- Clear sections using line breaks (not markdown)
- Connect ideas across domains
- Maintain clarity

You are responding to voice input on a visual canvas."""
    }

    return depth_instructions.get(depth_mode, depth_instructions['mirror'])
