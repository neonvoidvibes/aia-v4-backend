"""
Canvas view streaming routes.
Handles PTT-to-LLM streaming for the canvas interface.
"""

import os
import json
import logging
from flask import jsonify, Response, stream_with_context, g
from gotrue.types import User as SupabaseUser
from anthropic import AnthropicError
from tenacity import RetryError
from utils.s3_utils import get_latest_system_prompt

logger = logging.getLogger(__name__)


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

        if not anthropic_client:
            logger.error("Canvas stream fail: Anthropic client not initialized.")
            return jsonify({"error": "AI service unavailable"}), 503

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
        model_selection = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-5-20250929")
        temperature = 0.7

        def generate_canvas_stream():
            """Generator for canvas-specific streaming responses."""
            try:
                logger.info(f"Canvas stream started for Agent: {agent_name}, User: {user.id}, Depth: {depth_mode}")

                # Load agent-specific system prompt from S3
                agent_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
                logger.info(f"Loaded agent system prompt for '{agent_name}' ({len(agent_system_prompt)} chars)")

                # Combine agent prompt with canvas-specific depth instructions
                depth_instructions = get_canvas_depth_instructions(depth_mode)
                system_prompt = f"{agent_system_prompt}\n\n{depth_instructions}"

                # Simple message structure for canvas mode
                messages = [
                    {"role": "user", "content": transcript_text}
                ]

                logger.info(f"Calling Anthropic API for canvas stream (model: {model_selection}, depth: {depth_mode})")

                # Stream from Anthropic
                with anthropic_client.messages.stream(
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

RESPONSE STYLE:
- Provide concise, visually-impactful responses
- Be direct and insightful
- Use clear, powerful language suitable for large display
- Keep responses focused and structured
- Emphasize key insights with strong, declarative statements
- Avoid unnecessary preamble or filler
- Format for visual clarity when appropriate

You are responding to voice input, so be conversational yet impactful.""",

        'lens': """=== CANVAS MODE: LENS ===
You are now responding in Canvas mode with LENS depth.

RESPONSE STYLE:
- Provide structured, insightful analysis
- Break down complex ideas systematically
- Use clear headings and structure for visual scanning
- Provide deeper analysis while remaining concise
- Highlight connections and patterns
- Balance depth with clarity
- Format for visual hierarchy

You are responding to voice input. Provide thoughtful analysis in a visually digestible format.""",

        'portal': """=== CANVAS MODE: PORTAL ===
You are now responding in Canvas mode with PORTAL depth.

RESPONSE STYLE:
- Provide comprehensive, multifaceted responses
- Explore multiple perspectives and dimensions
- Use rich structure with clear sections
- Provide comprehensive insights while maintaining clarity
- Connect ideas across domains
- Balance breadth with focus
- Create visual rhythm with varied paragraph lengths

You are responding to voice input. Offer expansive thinking in an engaging, visual format."""
    }

    return depth_instructions.get(depth_mode, depth_instructions['mirror'])
