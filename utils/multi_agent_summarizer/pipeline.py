import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
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
        mirror_f = pool.submit(MirrorAgent().run, segments=segs)
        lens_f = pool.submit(lambda: LensAgent().run(segments=segs, mirror=mirror_f.result()))
        portal_f = pool.submit(lambda: PortalAgent().run(mirror=mirror_f.result(), lens=lens_f.result()))

    mirror_out = mirror_f.result()
    layer1 = mirror_out.get("layer1", {})
    mirror_level = mirror_out.get("mirror_level", {})

    layer2 = {
        "mirror_level": mirror_level,
        "lens_level": lens_f.result().get("lens_level", {}),
        "portal_level": portal_f.result().get("portal_level", {}),
    }

    wisdom = WisdomAgent().run(layer1=layer1, layer2=layer2)
    learn = LearningAgent().run(layer1=layer1, layer2=layer2, layer3=wisdom)

    integ = IntegrationAgent().run(layer1=layer1, layer2=layer2, layer3=wisdom, layer4=learn)

    full = FullSummary(**integ).dict()
    return full

