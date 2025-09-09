import json
import logging
from typing import Dict, Any, List
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import INTEGRATION_SYS
from utils.groq_client import integ_model


logger = logging.getLogger(__name__)


def _index_mirror(mirror_level: Dict[str, Any]) -> set:
    refs = set()
    for k in ["explicit_themes", "stated_agreements", "direct_quotes"]:
        items = (mirror_level or {}).get(k, []) or []
        for idx, _ in enumerate(items):
            refs.add(f"mirror:{k}:{idx}")
    return refs


def _index_lens(lens_level: Dict[str, Any]) -> set:
    refs = set()
    for k in ["hidden_patterns", "unspoken_tensions", "paradoxes_contradictions"]:
        items = (lens_level or {}).get(k, []) or []
        for idx, _ in enumerate(items):
            refs.add(f"lens:{k}:{idx}")
    return refs


class IntegrationAgent(Agent):
    name = "integration"

    def run(self, layer1: Dict[str, Any], layer2: Dict[str, Any], layer3: Dict[str, Any], layer4: Dict[str, Any]) -> Dict[str, Any]:
        # Attempt LLM integration first
        payload = {"layer1": layer1, "layer2": layer2, "layer3": layer3, "layer4": layer4}
        messages = [
            {"role": "system", "content": INTEGRATION_SYS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            resp = chat(integ_model(), messages, max_tokens=3200, temperature=0.1)
            data = safe_json_parse(resp)
            if isinstance(data, dict) and all(k in data for k in ["layer1", "layer2", "layer3", "layer4", "confidence"]):
                return data
        except Exception as e:
            logger.error(f"IntegrationAgent LLM error: {e}")

        # Deterministic local validation fallback
        mirror_level = (layer2 or {}).get("mirror_level") or {}
        lens_level = (layer2 or {}).get("lens_level") or {}
        portal_level = (layer2 or {}).get("portal_level") or {}

        mirror_refs = _index_mirror(mirror_level)
        lens_refs = _index_lens(lens_level)

        # Filter lens items without mirror evidence_ids
        def _filter_lens_items(items: List[Dict[str, Any]]):
            out = []
            for it in items or []:
                eids = set((it or {}).get("evidence_ids", []) or [])
                if not eids or not any(e.startswith("mirror:") for e in eids):
                    continue
                if not eids.intersection(mirror_refs):
                    continue
                out.append(it)
            return out

        lens_level["hidden_patterns"] = _filter_lens_items(lens_level.get("hidden_patterns", []))
        lens_level["unspoken_tensions"] = _filter_lens_items(lens_level.get("unspoken_tensions", []))
        lens_level["paradoxes_contradictions"] = _filter_lens_items(lens_level.get("paradoxes_contradictions", []))

        # Filter portal items without lens evidence
        def _filter_portal_items(items: List[Dict[str, Any]]):
            out = []
            for it in items or []:
                eids = set((it or {}).get("evidence_ids", []) or [])
                if not eids or not any(e.startswith("lens:") for e in eids):
                    continue
                if not eids.intersection(lens_refs):
                    continue
                out.append(it)
            return out

        portal_level["emergent_possibilities"] = _filter_portal_items(portal_level.get("emergent_possibilities", []))
        portal_level["intervention_opportunities"] = _filter_portal_items(portal_level.get("intervention_opportunities", []))
        portal_level["paradigm_shifts"] = _filter_portal_items(portal_level.get("paradigm_shifts", []))

        # Confidence heuristic
        conf = {
            "layer1": 0.7 if layer1 else 0.3,
            "layer2": 0.7 if (lens_level and portal_level) else 0.4,
            "layer3": 0.6 if layer3 else 0.3,
            "layer4": 0.6 if layer4 else 0.3,
        }

        return {
            "layer1": layer1,
            "layer2": {"mirror_level": mirror_level, "lens_level": lens_level, "portal_level": portal_level},
            "layer3": layer3,
            "layer4": layer4,
            "confidence": conf,
        }

