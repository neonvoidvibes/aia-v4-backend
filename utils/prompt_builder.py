import os
import logging
from typing import Optional, Dict, Any, List

from .s3_utils import (
    get_latest_system_prompt,
    get_latest_frameworks,
    get_latest_context,
    get_objective_function,
)

logger = logging.getLogger(__name__)


def _identity_header(agent: str, event: str) -> str:
    if event and event != '0000':
        return (
            f"You are the assistant for {agent}.\n"
            f"If event != '0000', you act for the {event} sub-team.\n"
            f"Primary: answer from {event}. If insufficient, use shared {agent} (0000).\n"
            f"Only pull other sub-teams if directly relevant; label such context explicitly."
        )
    return (
        f"You are the assistant for {agent}.\n"
        f"Primary: answer from shared {agent} (0000) materials."
    )


def _memory_scope_policy(agent: str, event: str) -> str:
    return (
        "When answering:\n"
        f"1) Prefer {event} materials.\n"
        f"2) Use {agent}/0000 as shared context if needed.\n"
        "3) Only use other events if directly relevant; state which event they came from.\n"
        f"Never let shared context override {event} norms unless user asks for cross-team input."
    )


def _rag_routing_policy(agent: str, event: str) -> str:
    return (
        f"Query up to 15 chunks:\n"
        f"- 7 from {event}, 5 from shared 0000, 3 from other events (may be 0).\n"
        f"Re-rank with MMR. Drop overflow to fit context.\n"
        f"Label non-event sources inline: [shared-0000] or [other:{{event_id}}]"
    )


def prompt_builder(
    agent: str,
    event: str = '0000',
    user_context: Optional[str] = None,
    retrieval_hints: Optional[Dict[str, Any]] = None,
    feature_event_prompts: bool = True,
) -> str:
    """
    Build a unified system prompt with layers:
    1. Identity header
    2. BASE: systemprompt_base + frameworks_base
    3. AGENT: systemprompt_aID-{agent} + context_aID-{agent}
    4. EVENT: systemprompt_eID-{event} + context_eID-{event}
    5. OBJECTIVE: event override -> agent -> base
    6. MEMORY_SCOPE_POLICY
    7. RAG_ROUTING_POLICY
    """

    parts: List[str] = []

    # 1) Identity
    parts.append("=== IDENTITY HEADER ===\n" + _identity_header(agent, event) + "\n=== END IDENTITY HEADER ===")

    # 2) Core Directive (if needed for wizard mode)
    # This will be handled in api_server.py for wizard mode
    
    # 3) Base System Prompt
    base_system_prompt = get_latest_system_prompt(None) or "You are a helpful assistant."
    parts.append("=== BASE SYSTEM PROMPT ===\n" + base_system_prompt + "\n=== END BASE SYSTEM PROMPT ===")

    # 4) Agent layer
    agent_system_prompt = get_latest_system_prompt(agent) or ""
    agent_context = get_latest_context(agent, None) or ""
    if agent_system_prompt or agent_context:
        block = ["=== AGENT LAYER ==="]
        if agent_system_prompt:
            block += ["-- system --", agent_system_prompt]
        if agent_context:
            block += ["-- context --", agent_context]
        block += ["=== END AGENT LAYER ==="]
        parts.append("\n".join(block))

    # 5) Event layer (optional / feature-flagged)
    if feature_event_prompts and event and event != '0000':
        # Support both legacy and new file patterns by reusing get_latest_context
        event_context = get_latest_context(agent, event) or ""

        # Try to load an event-specific prompt using legacy agent-based helper by convention:
        # New structure would be organizations/river/agents/{agent}/events/{event}/_config/systemprompt_eID-{event}
        # We piggyback get_latest_system_prompt(None) only for base; no agent+event helper exists, so keep context-only
        event_system_prompt = None
        try:
            # Attempt to fetch using s3_utils.find_file_any_extension via direct pattern
            from .s3_utils import get_cached_s3_file, find_file_any_extension
            event_pattern = f"organizations/river/agents/{agent}/events/{event}/_config/systemprompt_eID-{event}"
            event_system_prompt = get_cached_s3_file(
                cache_key=event_pattern,
                description=f"event system prompt for {agent}/{event}",
                fetch_function=lambda: find_file_any_extension(event_pattern, "event system prompt"),
            )
        except Exception:
            event_system_prompt = None

        if event_system_prompt or event_context:
            block = ["=== EVENT LAYER ==="]
            if event_system_prompt:
                block += ["-- system --", event_system_prompt]
            if event_context:
                block += ["-- context --", event_context]
            block += ["=== END EVENT LAYER ==="]
            parts.append("\n".join(block))

    # 5) Objective (agent override -> base). Event override supported by direct read if present
    # Try event-specific objective
    event_objective = None
    if feature_event_prompts and event and event != '0000':
        try:
            from .s3_utils import get_cached_s3_file, find_file_any_extension
            event_obj_pattern = f"organizations/river/agents/{agent}/events/{event}/_config/objective_function_eID-{event}"
            event_objective = get_cached_s3_file(
                cache_key=event_obj_pattern,
                description=f"event objective for {agent}/{event}",
                fetch_function=lambda: find_file_any_extension(event_obj_pattern, "event objective function"),
            )
        except Exception:
            event_objective = None

    objective = event_objective or get_objective_function(agent) or get_objective_function(None)
    if objective:
        parts.append("=== OBJECTIVE FUNCTION ===\n" + objective + "\n=== END OBJECTIVE FUNCTION ===")

    # 7) Base Frameworks (moved after objective function per new taxonomy)
    base_frameworks = get_latest_frameworks(None) or ""
    if base_frameworks:
        parts.append("=== BASE FRAMEWORKS ===\n" + base_frameworks + "\n=== END BASE FRAMEWORKS ===")

    # 8) Memory scope policy
    parts.append("=== MEMORY_SCOPE_POLICY ===\n" + _memory_scope_policy(agent, event) + "\n=== END MEMORY_SCOPE_POLICY ===")

    # 9) RAG routing policy
    parts.append("=== RAG_ROUTING_POLICY ===\n" + _rag_routing_policy(agent, event) + "\n=== END RAG_ROUTING_POLICY ===")

    # Note: Dynamic content sections (10-20) will be added by api_server.py
    # Note: USER CONTEXT (21) and CURRENT TIME (22) will be added by api_server.py at the end

    final_prompt = "\n\n".join([p for p in parts if p and p.strip()])
    logger.info(f"PromptBuilder: Built prompt for agent='{agent}', event='{event}'. Length={len(final_prompt)}")
    return final_prompt
    
    
def get_user_context_section(user_context: Optional[str]) -> str:
    """Generate USER CONTEXT section for api_server.py to append at the end."""
    if user_context:
        return f"=== USER CONTEXT ===\n{user_context}\n=== END USER CONTEXT ==="
    return ""

