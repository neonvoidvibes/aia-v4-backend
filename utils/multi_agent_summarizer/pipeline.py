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
from .feedback_parser import parse_reality_check_feedback


def summarize_transcript(agent_name: str, event_id: str, transcript_text: str, source_identifier: str) -> Dict:
    """
    Two-pass business-first pipeline:
    Pass 1: Initial analysis (Segmentation → Context → Business Reality → Org Dynamics → Strategic → Next Actions → Reality Check)
    Pass 2: Refinement based on Reality Check feedback → Final Integration
    """
    # Use the two-pass pipeline implementation
    results = run_pipeline_steps(transcript_text)
    
    # Return markdown-based summary (final integrated output)
    return {"__markdown__": results["full_md"]}


def run_pipeline_steps(transcript_text: str) -> Dict[str, Any]:
    """Two-pass pipeline with Reality Check feedback refinement.
    
    Pass 1: Initial analysis by all agents
    Pass 2: Refinement based on Reality Check feedback
    """
    # Step 1: Segmentation
    segs = SegmentationAgent().run(text=transcript_text)
    
    # PASS 1: Initial Analysis
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
    
    # PASS 2: Refinement based on Reality Check feedback
    feedback = parse_reality_check_feedback(reality_check_md)
    
    # Refine each agent's output if feedback exists
    if feedback.get("context"):
        context_md = ContextAgent().refine(segs, context_md, feedback["context"])
    
    if feedback.get("business_reality"):
        business_reality_md = BusinessRealityAgent().refine(segs, business_reality_md, 
                                                          feedback["business_reality"], context_md)
    
    if feedback.get("organizational_dynamics"):
        org_dynamics_md = OrganizationalDynamicsAgent().refine(segs, business_reality_md, 
                                                             org_dynamics_md, feedback["organizational_dynamics"], 
                                                             context_md)
    
    if feedback.get("strategic_implications"):
        strategic_md = StrategicImplicationsAgent().refine(business_reality_md, org_dynamics_md, 
                                                         strategic_md, feedback["strategic_implications"], 
                                                         context_md)
    
    if feedback.get("next_actions"):
        next_actions_md = NextActionsAgent().refine(business_reality_md, org_dynamics_md, 
                                                  strategic_md, next_actions_md, 
                                                  feedback["next_actions"], context_md)
    
    # Step 8: Final Integration (using refined outputs)
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