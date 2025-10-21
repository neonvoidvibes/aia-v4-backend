"""
Concurrent Canvas Analysis Execution - Global analysis generation for all agents.

This module provides utilities for running MLP analyses (mirror/lens/portal) concurrently
across all agents globally, with proper rate limiting and error handling.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone

from .canvas_analysis_agents import get_or_generate_analysis_doc
from .groq_rate_limiter import get_groq_rate_limiter

logger = logging.getLogger(__name__)


def run_concurrent_mlp_analysis(
    agent_name: str,
    event_id: str,
    force_refresh: bool = False,
    clear_previous: bool = False,
    transcript_listen_mode: str = 'latest',
    groups_read_mode: str = 'none',
    individual_raw_transcript_toggle_states: Optional[Dict[str, bool]] = None,
    saved_transcript_memory_mode: str = 'none',
    individual_memory_toggle_states: Optional[Dict[str, bool]] = None,
    event_type: str = 'shared',
    personal_layer: Optional[str] = None,
    personal_event_id: Optional[str] = None,
    allowed_events: Optional[Set] = None,
    event_types_map: Optional[Dict[str, str]] = None,
    event_profile: Optional[Dict[str, Any]] = None,
    max_workers: int = 3
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Run all three MLP analyses (mirror, lens, portal) concurrently for a single agent.

    This function coordinates parallel execution of all three analysis modes with:
    - Thread pool for concurrent execution
    - Shared rate limiter for Groq API compliance
    - Individual error handling per mode (failures don't block other modes)
    - Proper logging and result aggregation

    Args:
        agent_name: Agent name
        event_id: Event ID
        force_refresh: Force regeneration of analyses
        clear_previous: Clear previous analysis versions
        transcript_listen_mode: Transcript listen mode
        groups_read_mode: Groups read mode
        individual_raw_transcript_toggle_states: Toggle states for 'some' mode
        saved_transcript_memory_mode: Memorized transcript mode ('none'|'some'|'all')
        individual_memory_toggle_states: Toggle states for memorized transcript 'some' mode
        event_type: Event type
        personal_layer: Personal layer content
        personal_event_id: Personal event ID
        allowed_events: Set of allowed event IDs
        event_types_map: Event type mapping
        event_profile: Event profile metadata
        max_workers: Maximum worker threads (default: 3, one per mode)

    Returns:
        Dictionary mapping mode -> (current_doc, previous_doc)
        {
            'mirror': (current_str, previous_str),
            'lens': (current_str, previous_str),
            'portal': (current_str, previous_str)
        }

    Example:
        >>> results = run_concurrent_mlp_analysis(
        ...     agent_name="my_agent",
        ...     event_id="0000",
        ...     force_refresh=True
        ... )
        >>> mirror_current, mirror_previous = results['mirror']
    """
    start_time = datetime.now(timezone.utc)
    logger.info(
        f"Starting concurrent MLP analysis for {agent_name}/{event_id} "
        f"(force_refresh={force_refresh}, modes=['mirror', 'lens', 'portal'])"
    )

    # Get rate limiter stats for logging
    rate_limiter = get_groq_rate_limiter()
    initial_stats = rate_limiter.get_stats()
    logger.info(f"Rate limiter stats before analysis: {initial_stats}")

    def analyze_single_mode(mode: str) -> Tuple[str, Tuple[Optional[str], Optional[str]]]:
        """Helper function to analyze a single mode."""
        mode_start = datetime.now(timezone.utc)
        logger.info(f"[{mode}] Starting analysis for {agent_name}/{event_id}")

        try:
            current_doc, previous_doc = get_or_generate_analysis_doc(
                agent_name=agent_name,
                event_id=event_id,
                depth_mode=mode,
                force_refresh=force_refresh,
                clear_previous=clear_previous,
                transcript_listen_mode=transcript_listen_mode,
                groups_read_mode=groups_read_mode,
                individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states,
                saved_transcript_memory_mode=saved_transcript_memory_mode,
                individual_memory_toggle_states=individual_memory_toggle_states,
                event_type=event_type,
                personal_layer=personal_layer,
                personal_event_id=personal_event_id,
                allowed_events=allowed_events,
                event_types_map=event_types_map,
                event_profile=event_profile
            )

            duration = (datetime.now(timezone.utc) - mode_start).total_seconds()

            if current_doc:
                logger.info(
                    f"[{mode}] Analysis completed for {agent_name}/{event_id} "
                    f"({len(current_doc)} chars, {duration:.1f}s)"
                )
            else:
                logger.warning(
                    f"[{mode}] Analysis returned None for {agent_name}/{event_id} "
                    f"(likely due to rate limits or errors, {duration:.1f}s)"
                )

            return mode, (current_doc, previous_doc)

        except Exception as e:
            duration = (datetime.now(timezone.utc) - mode_start).total_seconds()
            logger.error(
                f"[{mode}] Analysis failed for {agent_name}/{event_id} ({duration:.1f}s): {e}",
                exc_info=True
            )
            return mode, (None, None)

    # Execute all three modes concurrently
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all three analysis tasks
        futures: Dict[Future, str] = {
            executor.submit(analyze_single_mode, mode): mode
            for mode in ['mirror', 'lens', 'portal']
        }

        # Collect results as they complete
        for future in as_completed(futures):
            mode = futures[future]
            try:
                result_mode, docs = future.result()
                results[result_mode] = docs
            except Exception as e:
                logger.error(f"[{mode}] Unexpected error collecting result: {e}", exc_info=True)
                results[mode] = (None, None)

    # Calculate total duration and log summary
    total_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    final_stats = rate_limiter.get_stats()

    successful_modes = [mode for mode, (current, _) in results.items() if current is not None]
    failed_modes = [mode for mode, (current, _) in results.items() if current is None]

    logger.info(
        f"Concurrent MLP analysis completed for {agent_name}/{event_id}: "
        f"{len(successful_modes)}/3 succeeded ({', '.join(successful_modes) if successful_modes else 'none'}), "
        f"{len(failed_modes)}/3 failed ({', '.join(failed_modes) if failed_modes else 'none'}), "
        f"duration={total_duration:.1f}s"
    )
    logger.info(f"Rate limiter stats after analysis: {final_stats}")

    return results


def run_global_mlp_analysis_for_all_agents(
    agents: List[str],
    event_id: str = '0000',
    force_refresh: bool = False,
    max_concurrent_agents: int = 3
) -> Dict[str, Dict[str, Tuple[Optional[str], Optional[str]]]]:
    """
    Run MLP analysis for multiple agents globally, processing agents concurrently.

    This is useful for batch operations like:
    - Nightly regeneration of all analyses
    - Initial setup for new deployment
    - Global refresh after system updates

    Args:
        agents: List of agent names to process
        event_id: Event ID (default: '0000')
        force_refresh: Force regeneration of all analyses
        max_concurrent_agents: Max agents to process simultaneously (default: 3)
                               Note: Each agent spawns 3 analysis threads (one per mode)

    Returns:
        Dictionary mapping agent_name -> mode -> (current_doc, previous_doc)
        {
            'agent1': {
                'mirror': (current, previous),
                'lens': (current, previous),
                'portal': (current, previous)
            },
            'agent2': { ... }
        }

    Example:
        >>> results = run_global_mlp_analysis_for_all_agents(
        ...     agents=['agent1', 'agent2', 'agent3'],
        ...     force_refresh=True,
        ...     max_concurrent_agents=2
        ... )
        >>> agent1_mirror = results['agent1']['mirror']
    """
    start_time = datetime.now(timezone.utc)
    logger.info(
        f"Starting global MLP analysis for {len(agents)} agents "
        f"(max_concurrent_agents={max_concurrent_agents}, force_refresh={force_refresh})"
    )

    def process_agent(agent_name: str) -> Tuple[str, Dict[str, Tuple[Optional[str], Optional[str]]]]:
        """Process all analyses for a single agent."""
        agent_start = datetime.now(timezone.utc)
        logger.info(f"[GLOBAL] Starting analysis for agent: {agent_name}")

        try:
            agent_results = run_concurrent_mlp_analysis(
                agent_name=agent_name,
                event_id=event_id,
                force_refresh=force_refresh,
                max_workers=3  # Each agent uses 3 workers (one per mode)
            )

            duration = (datetime.now(timezone.utc) - agent_start).total_seconds()
            successful_count = sum(1 for _, (current, _) in agent_results.items() if current is not None)

            logger.info(
                f"[GLOBAL] Completed analysis for agent: {agent_name} "
                f"({successful_count}/3 modes succeeded, {duration:.1f}s)"
            )

            return agent_name, agent_results

        except Exception as e:
            duration = (datetime.now(timezone.utc) - agent_start).total_seconds()
            logger.error(
                f"[GLOBAL] Failed to process agent: {agent_name} ({duration:.1f}s): {e}",
                exc_info=True
            )
            return agent_name, {
                'mirror': (None, None),
                'lens': (None, None),
                'portal': (None, None)
            }

    # Process agents concurrently
    global_results = {}
    with ThreadPoolExecutor(max_workers=max_concurrent_agents) as executor:
        futures: Dict[Future, str] = {
            executor.submit(process_agent, agent): agent
            for agent in agents
        }

        for future in as_completed(futures):
            agent = futures[future]
            try:
                agent_name, agent_results = future.result()
                global_results[agent_name] = agent_results
            except Exception as e:
                logger.error(f"[GLOBAL] Unexpected error collecting results for {agent}: {e}", exc_info=True)
                global_results[agent] = {
                    'mirror': (None, None),
                    'lens': (None, None),
                    'portal': (None, None)
                }

    # Calculate summary statistics
    total_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    total_analyses = len(agents) * 3  # 3 modes per agent
    successful_analyses = sum(
        sum(1 for _, (current, _) in agent_results.items() if current is not None)
        for agent_results in global_results.values()
    )

    logger.info(
        f"[GLOBAL] Global MLP analysis completed: "
        f"{len(agents)} agents processed, "
        f"{successful_analyses}/{total_analyses} analyses succeeded, "
        f"duration={total_duration:.1f}s, "
        f"avg_per_agent={total_duration/len(agents):.1f}s"
    )

    # Log rate limiter final stats
    rate_limiter = get_groq_rate_limiter()
    final_stats = rate_limiter.get_stats()
    logger.info(f"[GLOBAL] Final rate limiter stats: {final_stats}")

    return global_results
