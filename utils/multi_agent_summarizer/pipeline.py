import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from utils.schemas.transcript_summary import FullSummary
from .segmentation_agent import SegmentationAgent
from .mirror_agent import MirrorAgent
from .lens_agent import LensAgent
from .portal_agent import PortalAgent
from .wisdom_agent import WisdomAgent
from .learning_agent import LearningAgent
from .integration_agent import IntegrationAgent


def summarize_transcript(agent_name: str, event_id: str, transcript_text: str, source_identifier: str) -> Dict:
    segs = SegmentationAgent().run(text=transcript_text)

    with ThreadPoolExecutor(max_workers=4) as pool:
        mirror_md_f = pool.submit(MirrorAgent().run, segments=segs)
        lens_md_f = pool.submit(LensAgent().run, segments=[{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min"), "text": s.get("text")} for s in segs])
        portal_md_f = pool.submit(PortalAgent().run, segments=[{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min"), "text": s.get("text")} for s in segs])
        wisdom_md_f = pool.submit(WisdomAgent().run, segments=[{"id": s.get("id"), "text": s.get("text")} for s in segs])

    mirror_md = mirror_md_f.result()
    lens_md = lens_md_f.result()
    portal_md = portal_md_f.result()
    learning_md = LearningAgent().run(segments=[{"id": s.get("id"), "text": s.get("text")} for s in segs])
    wisdom_md = wisdom_md_f.result()

    final_md = "\n\n".join([
        mirror_md.strip(),
        lens_md.strip(),
        portal_md.strip(),
        wisdom_md.strip(),
        learning_md.strip(),
    ])

    # Return a minimal dict; markdown carries the summary
    return {"__markdown__": final_md}


def run_pipeline_steps(transcript_text: str) -> Dict[str, Any]:
    """Runs all agents and returns intermediates for debugging/inspection.
    Returns keys: segments, mirror, lens, portal, layer3, layer4, full
    """
    segs = SegmentationAgent().run(text=transcript_text)
    mirror_md = MirrorAgent().run(segments=segs)
    lens_md = LensAgent().run(segments=[{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min"), "text": s.get("text")} for s in segs])
    portal_md = PortalAgent().run(segments=[{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min"), "text": s.get("text")} for s in segs])
    layer3_md = WisdomAgent().run(segments=[{"id": s.get("id"), "text": s.get("text")} for s in segs])
    layer4_md = LearningAgent().run(segments=[{"id": s.get("id"), "text": s.get("text")} for s in segs])
    final_md = "\n\n".join([mirror_md.strip(), lens_md.strip(), portal_md.strip(), layer3_md.strip(), layer4_md.strip()])
    return {
        "segments": segs,
        "mirror_md": mirror_md,
        "lens_md": lens_md,
        "portal_md": portal_md,
        "layer3_md": layer3_md,
        "layer4_md": layer4_md,
        "full": {},
        "full_md": final_md,
    }
