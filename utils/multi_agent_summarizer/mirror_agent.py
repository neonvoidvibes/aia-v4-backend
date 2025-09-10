import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import MIRROR_SYS
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class MirrorAgent(Agent):
    name = "mirror"

    def run(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Heuristic language hint (Swedish common words)
        sample_text = "\n".join([s.get("text", "")[:400] for s in segments])
        lang_hint = "sv" if any(tok in sample_text.lower() for tok in [" det ", " och ", " vi ", " är ", " ska "]) else "en"

        # Two-pass: metadata then content
        seg_min = min([s.get("start_min", 0) for s in segments] or [0])
        seg_max = max([s.get("end_min", 0) for s in segments] or [0])

        # Pass A: metadata only
        meta_user = {
            "language_hint": lang_hint,
            "task": "meeting_metadata_only",
            "segments": [{"id": s.get("id"), "text": s.get("text", "")[:4000]} for s in segments[:3]],
            "approx_duration_minutes": max(1, int(seg_max - seg_min)),
        }
        messages_meta = [
            {"role": "system", "content": "Return JSON only. No commentary. Source content may be Swedish; produce English schema."},
            {"role": "user", "content": json.dumps(meta_user, ensure_ascii=False)},
        ]
        meta = None
        try:
            resp_meta = chat(std_model(), messages_meta, max_tokens=600, temperature=0.1, response_format={"type": "json_object"})
            meta = safe_json_parse(resp_meta) or {}
        except Exception as e:
            logger.error(f"MirrorAgent meta LLM error: {e}")

        # Pass B: content extraction
        content_user = {
            "language_hint": lang_hint,
            "need": ["layer1.actionable_outputs", "mirror_level"],
            "segments": [{"id": s.get("id"), "text": s.get("text", "")} for s in segments],
            "min_items": {"themes": 3, "agreements": 2, "quotes": 2}
        }
        messages_content = [
            {"role": "system", "content": "Extract Layer1 actionable_outputs and Mirror level as strict JSON. No prose. Lists should have at least the requested minimal items when evidence exists."},
            {"role": "user", "content": json.dumps(content_user, ensure_ascii=False)},
        ]
        content = None
        try:
            resp_cnt = chat(std_model(), messages_content, max_tokens=1600, temperature=0.1, response_format={"type": "json_object"})
            content = safe_json_parse(resp_cnt) or {}
        except Exception as e:
            logger.error(f"MirrorAgent content LLM error: {e}")

        # Merge
        out: Dict[str, Any] = {"layer1": {"meeting_metadata": {"timestamp": "unknown", "duration_minutes": max(1, int(seg_max - seg_min)), "participants_count": 1, "meeting_type": "unspecified", "primary_purpose": "unspecified"}, "actionable_outputs": {"decisions": [], "tasks": [], "next_meetings": []}}, "mirror_level": {"explicit_themes": [], "stated_agreements": [], "direct_quotes": [], "participation_patterns": {"energy_shifts": [], "engagement_distribution": "unknown"}}}
        if isinstance(meta, dict) and meta.get("meeting_metadata"):
            out["layer1"]["meeting_metadata"] = {**out["layer1"]["meeting_metadata"], **meta.get("meeting_metadata", {})}
        if isinstance(content, dict):
            # adopt actionable_outputs + mirror_level if present
            ao = content.get("actionable_outputs") or content.get("layer1", {}).get("actionable_outputs")
            if ao:
                out["layer1"]["actionable_outputs"] = ao
            if content.get("mirror_level"):
                out["mirror_level"] = {**out["mirror_level"], **content.get("mirror_level")}

        # If still too empty, retry with force-fill hint on a reduced segment slice
        if not out["mirror_level"].get("explicit_themes") and segments:
            try:
                slim = {"language_hint": lang_hint, "need": ["mirror_level"], "segments": [{"id": segments[0].get("id"), "text": segments[0].get("text", "")[:6000]}], "min_items": {"themes": 2, "agreements": 1, "quotes": 1}, "force": True}
                resp_retry = chat(std_model(), [{"role": "system", "content": "Return JSON only. Fill at least minimal items if evidence exists."}, {"role": "user", "content": json.dumps(slim, ensure_ascii=False)}], max_tokens=900, temperature=0.2, response_format={"type": "json_object"})
                retry = safe_json_parse(resp_retry) or {}
                if retry.get("mirror_level"):
                    out["mirror_level"] = {**out["mirror_level"], **retry.get("mirror_level")}
            except Exception as e:
                logger.error(f"MirrorAgent retry error: {e}")

        return out
