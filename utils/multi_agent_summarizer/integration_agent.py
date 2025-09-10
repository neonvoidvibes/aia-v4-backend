import json
import logging
from typing import Dict, Any, List
from .base import Agent
from ._llm import chat, safe_json_parse
from .md_parser import parse_full_markdown
from .md_templates import make_confidence_md
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
            import time
            t0 = time.perf_counter()
            resp = chat(integ_model(), messages, max_tokens=3200, temperature=0.3, response_format={"type": "json_object"})
            data = safe_json_parse(resp)
            if isinstance(data, dict) and all(k in data for k in ["layer1", "layer2", "layer3", "layer4", "confidence"]):
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"tx.agent.done name=IntegrationAgent ms={dt:.1f} out_chars={len(resp or '')}")
                return _normalize_full(data)
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

        for key in ["hidden_patterns", "unspoken_tensions", "paradoxes_contradictions"]:
            before = len(lens_level.get(key, []) or [])
            lens_level[key] = _filter_lens_items(lens_level.get(key, []))
            after = len(lens_level.get(key, []) or [])
            dropped = max(0, before - after)
            if dropped:
                logger.info(f"tx.integrate.prune layer=lens key={key} dropped={dropped} kept={after}")

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

        for key in ["emergent_possibilities", "intervention_opportunities", "paradigm_shifts"]:
            before = len(portal_level.get(key, []) or [])
            portal_level[key] = _filter_portal_items(portal_level.get(key, []))
            after = len(portal_level.get(key, []) or [])
            dropped = max(0, before - after)
            if dropped:
                logger.info(f"tx.integrate.prune layer=portal key={key} dropped={dropped} kept={after}")

        # Confidence heuristic
        conf = {
            "layer1": 0.7 if layer1 else 0.3,
            "layer2": 0.7 if (lens_level and portal_level) else 0.4,
            "layer3": 0.6 if layer3 else 0.3,
            "layer4": 0.6 if layer4 else 0.3,
        }
        logger.info(
            "tx.integrate.done conf=" + 
            ",".join([f"{k}={v:.2f}" for k,v in conf.items()])
        )

        return _normalize_full({
            "layer1": layer1,
            "layer2": {"mirror_level": mirror_level, "lens_level": lens_level, "portal_level": portal_level},
            "layer3": layer3,
            "layer4": layer4,
            "confidence": conf,
        })

    def run_md(self, *, context_md: str, business_reality_md: str, organizational_dynamics_md: str, 
               strategic_implications_md: str, reality_check_md: str) -> str:
        """New business-first integration that combines all layers into executive summary.
        Returns final markdown output focused on business value.
        """
        
        payload = {
            "context_content": context_md[:2000],
            "business_reality_content": business_reality_md[:5000],  # Increased since it now includes next actions
            "organizational_dynamics_content": organizational_dynamics_md[:3000],
            "strategic_implications_content": strategic_implications_md[:3000], 
            "reality_check_content": reality_check_md[:2000]
        }
        
        try:
            from ._llm import chat
            from utils.groq_client import std_model
            
            integration = chat(
                std_model(),
                [
                    {"role": "system", "content": INTEGRATION_SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=3000,
                temperature=0.3,
            )
            
            return integration or "# Executive Summary\n(Integration failed)\n"
            
        except Exception as e:
            logger.error(f"IntegrationAgent run_md error: {e}")
            # Fallback: simple concatenation
            parts = [
                context_md.strip(),
                business_reality_md.strip(),  # Now includes next actions
                organizational_dynamics_md.strip(),
                strategic_implications_md.strip(),
                reality_check_md.strip()
            ]
            return "\n\n".join([p for p in parts if p])

    # Legacy method for backward compatibility
    def run_md_legacy(self, *, layer1_md: str, layer2_mirror_md: str, layer2_lens_md: str, layer2_portal_md: str, layer3_md: str, layer4_md: str) -> Dict[str, Any]:
        parts = [
            layer1_md.strip(), layer2_mirror_md.strip(), layer2_lens_md.strip(), layer2_portal_md.strip(), layer3_md.strip(), layer4_md.strip()
        ]
        combined_md = "\n\n".join([p for p in parts if p])
        parsed = parse_full_markdown(combined_md)
        conf = {
            'layer1': 0.2 + 0.2*bool(parsed.get('layer1', {}).get('meeting_metadata')) + 0.2*bool(parsed.get('layer1', {}).get('actionable_outputs', {}).get('decisions')),
            'layer2': 0.2 + 0.2*bool(parsed.get('layer2', {}).get('mirror_level', {}).get('explicit_themes')) + 0.2*bool(parsed.get('layer2', {}).get('lens_level', {}).get('hidden_patterns')) + 0.2*bool(parsed.get('layer2', {}).get('portal_level', {}).get('emergent_possibilities')),
            'layer3': 0.2 + 0.2*bool(parsed.get('layer3', {}).get('connectedness_patterns', {}).get('self_awareness')),
            'layer4': 0.2 + 0.2*bool(parsed.get('layer4', {}).get('triple_loop_learning', {}).get('single_loop')),
        }
        parsed['confidence'] = conf
        parsed['__markdown__'] = combined_md + "\n\n" + make_confidence_md(conf)
        logger.info("tx.integrate.done conf=" + ",".join([f"{k}={v:.2f}" for k,v in conf.items()]))
        return parsed


def _ensure_path(d: Dict[str, Any], path: List[str], default: Any) -> Any:
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    leaf = path[-1]
    if leaf not in cur or cur[leaf] is None:
        cur[leaf] = default
    return cur[leaf]


def _normalize_full(full: Dict[str, Any]) -> Dict[str, Any]:
    # Layer1 meeting_metadata defaults
    mm = _ensure_path(full, ["layer1"], {})
    mm = _ensure_path(full, ["layer1", "meeting_metadata"], {})
    if not isinstance(mm, dict):
        full["layer1"]["meeting_metadata"] = {}
        mm = full["layer1"]["meeting_metadata"]
    if not isinstance(full["layer1"].get("actionable_outputs"), dict):
        full["layer1"]["actionable_outputs"] = {"decisions": [], "tasks": [], "next_meetings": []}

    # Required strings
    for key, default in [
        ("timestamp", "unknown"),
        ("meeting_type", "unspecified"),
        ("primary_purpose", "unspecified"),
    ]:
        val = mm.get(key)
        if not isinstance(val, str):
            mm[key] = default
    # Required numeric bounds
    dm = mm.get("duration_minutes")
    if not isinstance(dm, int) or dm < 0:
        mm["duration_minutes"] = 0
    pc = mm.get("participants_count")
    if not isinstance(pc, int) or pc < 1:
        mm["participants_count"] = 1

    # Layer2 mirror_level participation_patterns
    mirror_level = full.get("layer2", {}).get("mirror_level")
    if not isinstance(mirror_level, dict):
        full.setdefault("layer2", {})["mirror_level"] = {}
        mirror_level = full["layer2"]["mirror_level"]
    pp = mirror_level.get("participation_patterns")
    if not isinstance(pp, dict):
        mirror_level["participation_patterns"] = {"energy_shifts": [], "engagement_distribution": "unknown"}
    else:
        if not isinstance(pp.get("energy_shifts"), list):
            pp["energy_shifts"] = []
        if not isinstance(pp.get("engagement_distribution"), str):
            pp["engagement_distribution"] = "unknown"

    # Layer2 lens_level group_dynamics
    lens_level = full.get("layer2", {}).get("lens_level")
    if not isinstance(lens_level, dict):
        full.setdefault("layer2", {})["lens_level"] = {}
        lens_level = full["layer2"]["lens_level"]
    gd = lens_level.get("group_dynamics")
    if not isinstance(gd, dict):
        lens_level["group_dynamics"] = {"emotional_undercurrents": "", "power_dynamics": ""}
    else:
        if not isinstance(gd.get("emotional_undercurrents"), str):
            gd["emotional_undercurrents"] = ""
        if not isinstance(gd.get("power_dynamics"), str):
            gd["power_dynamics"] = ""

    # Layer3 transcendental_alignment coherence_quality
    ta = full.get("layer3", {}).get("transcendental_alignment")
    if not isinstance(ta, dict):
        full.setdefault("layer3", {})["transcendental_alignment"] = {"beauty_moments": [], "truth_emergence": [], "goodness_orientation": [], "coherence_quality": ""}
    else:
        if not isinstance(ta.get("coherence_quality"), str):
            ta["coherence_quality"] = ""

    # Ensure lists exist where models expect lists
    def ensure_list(path: List[str]):
        cur = full
        for k in path[:-1]:
            cur = cur.setdefault(k, {}) if isinstance(cur, dict) else {}
        leaf = path[-1]
        if not isinstance(cur.get(leaf), list):
            cur[leaf] = []

    # Layer1 actionable outputs lists
    for p in [["layer1", "actionable_outputs", "decisions"], ["layer1", "actionable_outputs", "tasks"], ["layer1", "actionable_outputs", "next_meetings"]]:
        ensure_list(p)

    return full
