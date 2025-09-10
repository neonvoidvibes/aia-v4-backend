import json
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from utils.schemas.transcript_summary import FullSummary
from .segmentation_agent import SegmentationAgent
from .context_agent import ContextAgent
from .business_reality_agent import BusinessRealityAgent
from .organizational_dynamics_agent import OrganizationalDynamicsAgent
from .strategic_implications_agent import StrategicImplicationsAgent
from .reality_check_agent import RealityCheckAgent
from .wisdom_learning_agent import WisdomLearningAgent
from .integration_agent import IntegrationAgent
from .feedback_parser import parse_reality_check_feedback
from .repetition_detector import RepetitionDetector


def extract_datetime_from_filename(filename: str) -> Optional[str]:
    """
    Extract date/time from filename with format like:
    transcript_D20250812-T080755_uID-...
    Returns formatted datetime string or None if not found.
    """
    if not filename:
        return None
    
    # Extract date and time components
    match = re.search(r'_D(\d{8})-T(\d{6})_', filename)
    if not match:
        return None
    
    date_str, time_str = match.groups()
    
    try:
        # Parse components
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        # Create datetime object (UTC as specified)
        dt = datetime(year, month, day, hour, minute, second)
        
        # Format for display
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, IndexError):
        return None


def summarize_transcript(agent_name: str, event_id: str, transcript_text: str, source_identifier: str, filename: str = None) -> Dict:
    """
    Two-pass business-first pipeline:
    Pass 1: Initial analysis (Segmentation → Context → Business Reality → Org Dynamics → Strategic → Reality Check)
    Pass 2: Refinement based on Reality Check feedback → Final Integration
    """
    # Extract datetime from filename if provided
    meeting_datetime = None
    if filename:
        meeting_datetime = extract_datetime_from_filename(filename)
    
    # Use the two-pass pipeline implementation
    results = run_pipeline_steps(transcript_text, meeting_datetime=meeting_datetime)
    
    # Return markdown-based summary (final integrated output)
    return {"__markdown__": results["full_md"]}


def run_pipeline_steps(transcript_text: str, meeting_datetime: Optional[str] = None) -> Dict[str, Any]:
    """Two-pass pipeline with Reality Check feedback refinement.
    
    Pass 1: Initial analysis by all agents
    Pass 2: Refinement based on Reality Check feedback
    
    Args:
        transcript_text: The transcript content to analyze
        meeting_datetime: Optional datetime string from filename (e.g., "2025-08-12 08:07:55 UTC")
    """
    # Step 1: Segmentation
    segs = SegmentationAgent().run(text=transcript_text)
    
    # Step 2: Repetition Detection
    repetition_detector = RepetitionDetector()
    repetition_analysis = repetition_detector.detect_repetitions(segs)
    
    # PASS 1: Initial Analysis
    # Step 3: Business Context
    context_md = ContextAgent().run(segments=segs, repetition_analysis=repetition_analysis, meeting_datetime=meeting_datetime)

    # Step 4: Business Reality
    business_reality_md = BusinessRealityAgent().run(segments=segs, context_md=context_md, 
                                                      repetition_analysis=repetition_analysis, meeting_datetime=meeting_datetime)
    
    # Step 5: Organizational Dynamics (now has direct segment access)
    org_dynamics_md = OrganizationalDynamicsAgent().run(segs, business_reality_md, context_md, 
                                                         repetition_analysis=repetition_analysis, meeting_datetime=meeting_datetime)
    
    # Step 6: Strategic Implications (now has direct segment access)
    strategic_md = StrategicImplicationsAgent().run(segs, business_reality_md, org_dynamics_md, context_md, 
                                                     repetition_analysis=repetition_analysis, meeting_datetime=meeting_datetime)
    
    # Step 7: Wisdom and Learning
    wisdom_learning_md = WisdomLearningAgent().run(segs, context_md, business_reality_md,
                                                   org_dynamics_md, strategic_md, 
                                                   repetition_analysis=repetition_analysis, meeting_datetime=meeting_datetime)
    
    # Step 8: Reality Check
    reality_check_md = RealityCheckAgent().run(segs, context_md, business_reality_md, 
                                              org_dynamics_md, strategic_md,
                                              repetition_analysis=repetition_analysis, meeting_datetime=meeting_datetime)
    
    # PASS 2: Refinement based on Reality Check feedback
    feedback = parse_reality_check_feedback(reality_check_md)
    
    # Refine each agent's output if feedback exists
    if feedback.get("context"):
        context_md = ContextAgent().refine(segs, context_md, feedback["context"], repetition_analysis, meeting_datetime=meeting_datetime)
    
    if feedback.get("business_reality"):
        business_reality_md = BusinessRealityAgent().refine(segs, business_reality_md, 
                                                          feedback["business_reality"], context_md,
                                                          repetition_analysis, meeting_datetime=meeting_datetime)
    
    if feedback.get("organizational_dynamics"):
        org_dynamics_md = OrganizationalDynamicsAgent().refine(segs, business_reality_md, 
                                                             org_dynamics_md, feedback["organizational_dynamics"], 
                                                             context_md, repetition_analysis, meeting_datetime=meeting_datetime)
    
    if feedback.get("strategic_implications"):
        strategic_md = StrategicImplicationsAgent().refine(segs, business_reality_md, org_dynamics_md, 
                                                         strategic_md, feedback["strategic_implications"], 
                                                         context_md, repetition_analysis, meeting_datetime=meeting_datetime)
    
    if feedback.get("wisdom_learning"):
        wisdom_learning_md = WisdomLearningAgent().refine(segs, context_md, business_reality_md,
                                                         org_dynamics_md, strategic_md, 
                                                         wisdom_learning_md, feedback["wisdom_learning"],
                                                         repetition_analysis, meeting_datetime=meeting_datetime)
    
    # Step 8: Final Integration
    final_md = IntegrationAgent().run_md(
        context_md=context_md,
        business_reality_md=business_reality_md,
        organizational_dynamics_md=org_dynamics_md,
        strategic_implications_md=strategic_md,
        reality_check_md=reality_check_md
    )
    
    return {
        "segments": segs,
        "context_md": context_md,
        "business_reality_md": business_reality_md,
        "org_dynamics_md": org_dynamics_md,
        "strategic_md": strategic_md,
        "wisdom_learning_md": wisdom_learning_md,
        "reality_check_md": reality_check_md,
        "full_md": final_md,
    }