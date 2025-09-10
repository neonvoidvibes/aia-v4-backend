import json
import logging
from typing import List, Dict, Any
from .base import Agent
from ._llm import chat, safe_json_parse
from .prompts import MIRROR_SYS
from .md_templates import L1_HEADER, L2_HEADER, wrap_section, kv_line
from utils.groq_client import std_model


logger = logging.getLogger(__name__)


class MirrorAgent(Agent):
    name = "mirror"

    def run(self, segments: List[Dict[str, Any]]):
        """Map-Reduce Mirror producing Markdown sections for Layer1 metadata and Mirror level.
        Returns a markdown string of the two sections.
        """
        # Heuristic language hint (language-agnostic content, keys in English)
        sample_text = "\n".join([s.get("text", "")[:400] for s in segments])
        lang_hint = "auto"

        # Two-pass: metadata (tiny) then evidence (map->reduce)
        seg_min = min([s.get("start_min", 0) for s in segments] or [0])
        seg_max = max([s.get("end_min", 0) for s in segments] or [0])

        # Pass A is skipped to avoid duplication; reduce step will include Layer 1 metadata
        meta_md = ""

        # Map per-segment evidence packs
        packs: List[str] = []
        for s in segments:
            user = {
                "language_hint": lang_hint,
                "segment": {"id": s.get("id"), "start_min": s.get("start_min"), "end_min": s.get("end_min"), "text": s.get("text")},
                "template": """
## Mirror Pack {seg_id}
### Quotes
- quote: <verbatim> | significance: <why important> | segment_ids: {seg_id}
### Explicit Themes
- theme: <phrase> | frequency: <int> | segment_ids: {seg_id}
### Stated Agreements
- agreement: <phrase> | confidence: low|medium|high | segment_ids: {seg_id}
### Participation
- engagement_distribution: <string> | energy_shifts: <comma list>
"""
            }
            msg = [
                {"role": "system", "content": "Extract Mirror evidence for this single segment. Return Markdown only. Preserve original language in content; use English keys."},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ]
            try:
                pack = chat(std_model(), msg, max_tokens=1200, temperature=0.1)
                packs.append(pack)
            except Exception as e:
                logger.error(f"MirrorAgent pack error for {s.get('id')}: {e}")

        # Reduce: global Layer1 actionable + Mirror aggregation from packs only
        reduce_user = {
            "language_hint": lang_hint,
            "approx_duration_minutes": max(1, int(seg_max - seg_min)),
            "evidence_packs": packs,
            "template": """
# Layer 1 — Traditional Business
- timestamp: <ISO or 'unknown'>
- duration_minutes: <int>
- participants_count: <int>
- meeting_type: <string>
- primary_purpose: <string>

### Decisions
- decision: <text> | context: <text> | urgency: low|medium|high | source_span: seg:X:charStart-charEnd
### Tasks
- task: <text> | owner_role: <text> | deadline: YYYY-MM-DD | dependencies: a, b | source_span: seg:X:...
### Next Meetings
- purpose: <text> | timeframe: <text> | required_participants: a, b | source_span: seg:X:...

# Layer 2 — Collective Intelligence
### Mirror
- theme: <text> | context: <text> | frequency: <int> | segment_ids: seg:1,seg:2
- agreement: <text> | confidence: low|medium|high | segment_ids: seg:1
- quote: <verbatim> | context: <text> | significance: <text> | segment_ids: seg:1
- engagement_distribution: <string> | energy_shifts: a, b
"""
        }
        reduce_msg = [
            {"role": "system", "content": "Merge segment packs into Layer1 and Mirror sections. Markdown only. Keep keys in English, content in original language."},
            {"role": "user", "content": json.dumps(reduce_user, ensure_ascii=False)}
        ]
        try:
            reduced = chat(std_model(), reduce_msg, max_tokens=2000, temperature=0.1)
        except Exception as e:
            logger.error(f"MirrorAgent reduce error: {e}")
            reduced = ""

        md = []
        if reduced:
            md.append(reduced.strip())
        else:
            # minimal fallback
            md.append(L1_HEADER + "\n")
            md.append(kv_line({"timestamp": "unknown", "duration_minutes": max(1, int(seg_max - seg_min)), "participants_count": 1, "meeting_type": "unspecified", "primary_purpose": "unspecified"}))
            md.append("\n" + L2_HEADER + "\n### Mirror\n- engagement_distribution: unknown\n\n")

        return "\n".join([m for m in ["\n".join(md).strip()] if m])
