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

    with ThreadPoolExecutor(max_workers=3) as pool:
        mirror_md_f = pool.submit(MirrorAgent().run, segments=segs)
        mirror_md = mirror_md_f.result()
        # Lens depends on mirror_md
        lens_md = LensAgent().run(segments=[{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min")} for s in segs], mirror_markdown=mirror_md)
        # Portal depends on mirror_md + lens_md
        portal_md = PortalAgent().run(mirror_markdown=mirror_md, lens_markdown=lens_md)

    # Wisdom and Learning
    wisdom_md = WisdomAgent().run(layer1_md=mirror_md, layer2_md="\n\n".join([mirror_md, lens_md, portal_md]))
    learning_md = LearningAgent().run(layer1_md=mirror_md, layer2_md="\n\n".join([mirror_md, lens_md, portal_md]), layer3_md=wisdom_md)

    # Integrate -> parse to dict + combined markdown
    integ = IntegrationAgent().run_md(
        layer1_md=mirror_md,
        layer2_mirror_md=mirror_md,
        layer2_lens_md=lens_md,
        layer2_portal_md=portal_md,
        layer3_md=wisdom_md,
        layer4_md=learning_md,
    )

    full = FullSummary(**{k: v for k, v in integ.items() if k in ['layer1','layer2','layer3','layer4','confidence']}).dict()
    full['__markdown__'] = integ.get('__markdown__', '')
    return full


def run_pipeline_steps(transcript_text: str) -> Dict[str, Any]:
    """Runs all agents and returns intermediates for debugging/inspection.
    Returns keys: segments, mirror, lens, portal, layer3, layer4, full
    """
    segs = SegmentationAgent().run(text=transcript_text)
    mirror_md = MirrorAgent().run(segments=segs)
    lens_md = LensAgent().run(segments=[{"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min")} for s in segs], mirror_markdown=mirror_md)
    portal_md = PortalAgent().run(mirror_markdown=mirror_md, lens_markdown=lens_md)
    layer3_md = WisdomAgent().run(layer1_md=mirror_md, layer2_md="\n\n".join([mirror_md, lens_md, portal_md]))
    layer4_md = LearningAgent().run(layer1_md=mirror_md, layer2_md="\n\n".join([mirror_md, lens_md, portal_md]), layer3_md=layer3_md)
    integ = IntegrationAgent().run_md(
        layer1_md=mirror_md,
        layer2_mirror_md=mirror_md,
        layer2_lens_md=lens_md,
        layer2_portal_md=portal_md,
        layer3_md=layer3_md,
        layer4_md=layer4_md,
    )
    full = FullSummary(**{k: v for k, v in integ.items() if k in ['layer1','layer2','layer3','layer4','confidence']}).dict()
    full['__markdown__'] = integ.get('__markdown__', '')
    return {
        "segments": segs,
        "mirror_md": mirror_md,
        "lens_md": lens_md,
        "portal_md": portal_md,
        "layer3_md": layer3_md,
        "layer4_md": layer4_md,
        "full": full,
        "full_md": integ.get('__markdown__', ''),
    }
