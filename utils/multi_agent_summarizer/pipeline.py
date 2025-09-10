import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from utils.schemas.transcript_summary import FullSummary
from .segmentation_agent import SegmentationAgent
from .context_agent import ContextAgent
from .business_reality_agent import BusinessRealityAgent
from .organizational_dynamics_agent import OrganizationalDynamicsAgent
from .strategic_implications_agent import StrategicImplicationsAgent
from .next_actions_agent import NextActionsAgent
from .reality_check_agent import RealityCheckAgent
from .integration_agent import IntegrationAgent


def summarize_transcript(agent_name: str, event_id: str, transcript_text: str, source_identifier: str) -> Dict:
    """
    Business-first pipeline:
    1. Segmentation
    2. Context (business context, not story)
    3. Parallel: Business Reality + Organizational Dynamics + Strategic Implications
    4. Next Actions (fed by all previous layers)
    5. Reality Check (validates all outputs)
    6. Integration (final synthesis)
    """
    # Step 1: Segmentation
    segs = SegmentationAgent().run(text=transcript_text)
    
    # Step 2: Business Context (replaces storyteller)
    context_md = ContextAgent().run(segments=segs)

    # Step 3: Parallel processing of core analysis layers
    with ThreadPoolExecutor(max_workers=3) as pool:
        business_reality_f = pool.submit(BusinessRealityAgent().run, segments=segs, context_md=context_md)
        org_dynamics_f = pool.submit(OrganizationalDynamicsAgent().run, "", context_md)  # Will fix after business reality
        strategic_f = pool.submit(StrategicImplicationsAgent().run, "", "", context_md)  # Will fix after org dynamics
    
    # Get business reality first
    business_reality_md = business_reality_f.result()
    
    # Now run org dynamics with business reality as input
    org_dynamics_md = OrganizationalDynamicsAgent().run(business_reality_md, context_md)
    
    # Now run strategic implications with both inputs
    strategic_md = StrategicImplicationsAgent().run(business_reality_md, org_dynamics_md, context_md)
    
    # Step 4: Next Actions (sequential, needs all previous layers)
    next_actions_md = NextActionsAgent().run(business_reality_md, org_dynamics_md, strategic_md, context_md)
    
    # Step 5: Reality Check (validates everything)
    reality_check_md = RealityCheckAgent().run(segs, context_md, business_reality_md, 
                                              org_dynamics_md, strategic_md, next_actions_md)
    
    # Step 6: Integration (final synthesis)
    final_md = IntegrationAgent().run_md(
        context_md=context_md,
        business_reality_md=business_reality_md,
        organizational_dynamics_md=org_dynamics_md,
        strategic_implications_md=strategic_md,
        next_actions_md=next_actions_md,
        reality_check_md=reality_check_md
    )

    # Return markdown-based summary
    return {"__markdown__": final_md}


def run_pipeline_steps(transcript_text: str) -> Dict[str, Any]:
    """Runs all agents and returns intermediates for debugging/inspection.
    New structure: context, business_reality, org_dynamics, strategic, next_actions, reality_check
    """
    # Step 1: Segmentation
    segs = SegmentationAgent().run(text=transcript_text)
    
    # Step 2: Business Context
    context_md = ContextAgent().run(segments=segs)

    # Step 3: Business Reality
    business_reality_md = BusinessRealityAgent().run(segments=segs, context_md=context_md)
    
    # Step 4: Organizational Dynamics
    org_dynamics_md = OrganizationalDynamicsAgent().run(business_reality_md, context_md)
    
    # Step 5: Strategic Implications  
    strategic_md = StrategicImplicationsAgent().run(business_reality_md, org_dynamics_md, context_md)
    
    # Step 6: Next Actions
    next_actions_md = NextActionsAgent().run(business_reality_md, org_dynamics_md, strategic_md, context_md)
    
    # Step 7: Reality Check
    reality_check_md = RealityCheckAgent().run(segs, context_md, business_reality_md, 
                                              org_dynamics_md, strategic_md, next_actions_md)
    
    # Step 8: Final Integration
    final_md = IntegrationAgent().run_md(
        context_md=context_md,
        business_reality_md=business_reality_md,
        organizational_dynamics_md=org_dynamics_md,
        strategic_implications_md=strategic_md,
        next_actions_md=next_actions_md,
        reality_check_md=reality_check_md
    )
    
    return {
        "segments": segs,
        "context_md": context_md,
        "business_reality_md": business_reality_md,
        "org_dynamics_md": org_dynamics_md,
        "strategic_md": strategic_md,
        "next_actions_md": next_actions_md,
        "reality_check_md": reality_check_md,
        "full_md": final_md,
    }