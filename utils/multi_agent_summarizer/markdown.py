from typing import Dict, Any, List


def _kv(k: str, v: Any) -> str:
    if v is None or v == "":
        return ""
    return f"- {k}: {v}\n"


def full_summary_to_markdown(full: Dict[str, Any]) -> str:
    lines: List[str] = []
    l1 = full.get("layer1", {})
    l2 = full.get("layer2", {})
    l3 = full.get("layer3", {})
    l4 = full.get("layer4", {})
    conf = full.get("confidence", {})

    # Header
    lines.append("# Transcript Summary\n")

    # Confidence
    if conf:
        lines.append("## Confidence\n")
        for k in ["layer1", "layer2", "layer3", "layer4"]:
            if k in conf:
                lines.append(f"- {k}: {conf[k]:.2f}\n")
        lines.append("\n")

    # Layer 1
    mm = (l1 or {}).get("meeting_metadata", {})
    ao = (l1 or {}).get("actionable_outputs", {})
    lines.append("## Layer 1 — Traditional Business\n")
    lines.append(_kv("timestamp", mm.get("timestamp")))
    lines.append(_kv("duration_minutes", mm.get("duration_minutes")))
    lines.append(_kv("participants_count", mm.get("participants_count")))
    lines.append(_kv("meeting_type", mm.get("meeting_type")))
    lines.append(_kv("primary_purpose", mm.get("primary_purpose")))
    lines.append("\n### Decisions\n")
    for d in ao.get("decisions", []) or []:
        lines.append(f"- {d.get('decision')}: {d.get('context')} (urgency: {d.get('urgency')})\n")
    lines.append("\n### Tasks\n")
    for t in ao.get("tasks", []) or []:
        deps = ", ".join(t.get("dependencies", []) or [])
        lines.append(f"- {t.get('task')} — owner: {t.get('owner_role')}, deadline: {t.get('deadline') or 'n/a'}, deps: {deps}\n")
    lines.append("\n### Next Meetings\n")
    for nm in ao.get("next_meetings", []) or []:
        req = ", ".join(nm.get("required_participants", []) or [])
        lines.append(f"- {nm.get('purpose')} — timeframe: {nm.get('timeframe')}, required: {req}\n")

    # Layer 2
    lines.append("\n## Layer 2 — Collective Intelligence\n")
    mirror = (l2 or {}).get("mirror_level", {})
    lens = (l2 or {}).get("lens_level", {})
    portal = (l2 or {}).get("portal_level", {})
    lines.append("### Mirror\n")
    for et in mirror.get("explicit_themes", []) or []:
        lines.append(f"- Theme: {et.get('theme')} (freq {et.get('frequency')}) — {et.get('context')}\n")
    for ag in mirror.get("stated_agreements", []) or []:
        lines.append(f"- Agreement: {ag.get('agreement')} (conf {ag.get('confidence')})\n")
    for dq in mirror.get("direct_quotes", []) or []:
        lines.append(f"- Quote: \"{dq.get('quote')}\" — {dq.get('significance')}\n")
    pp = mirror.get("participation_patterns", {})
    if pp:
        lines.append(_kv("engagement_distribution", pp.get("engagement_distribution")))
        if pp.get("energy_shifts"):
            lines.append(f"- energy_shifts: {', '.join(pp.get('energy_shifts'))}\n")
    lines.append("\n### Lens\n")
    for hp in lens.get("hidden_patterns", []) or []:
        lines.append(f"- Hidden: {hp.get('pattern')} — {hp.get('systemic_significance')}\n")
    for ut in lens.get("unspoken_tensions", []) or []:
        lines.append(f"- Tension: {ut.get('tension')} — {ut.get('impact_assessment')}\n")
    gd = lens.get("group_dynamics", {})
    if gd:
        lines.append(_kv("emotional_undercurrents", gd.get("emotional_undercurrents")))
        lines.append(_kv("power_dynamics", gd.get("power_dynamics")))
    for pz in lens.get("paradoxes_contradictions", []) or []:
        lines.append(f"- Paradox: {pz.get('paradox')} — {pz.get('implications')}\n")
    lines.append("\n### Portal\n")
    for ep in portal.get("emergent_possibilities", []) or []:
        lines.append(f"- Possibility: {ep.get('possibility')} — {ep.get('transformation_potential')}\n")
    for io in portal.get("intervention_opportunities", []) or []:
        lines.append(f"- Leverage: {io.get('leverage_point')} (p={io.get('probability_score')})\n")
    for ps in portal.get("paradigm_shifts", []) or []:
        lines.append(f"- Shift: {ps.get('shift')} — {ps.get('readiness_assessment')}\n")

    # Layer 3
    lines.append("\n## Layer 3 — Wisdom Integration\n")
    c = (l3 or {}).get("connectedness_patterns", {})
    lines.append("### Connectedness Patterns\n")
    for sa in c.get("self_awareness", []) or []:
        lines.append(f"- SA: {sa.get('insight')} — growth: {sa.get('individual_growth')}\n")
    for idt in c.get("interpersonal_dynamics", []) or []:
        lines.append(f"- ID: {idt.get('relationship_shift')} — collaboration: {idt.get('collaboration_quality')}\n")
    for su in c.get("systemic_understanding", []) or []:
        lines.append(f"- SU: {su.get('systems_insight')} — context: {su.get('broader_context')}\n")
    ta = (l3 or {}).get("transcendental_alignment", {})
    lines.append("\n### Transcendental Alignment\n")
    for bm in ta.get("beauty_moments", []) or []:
        lines.append(f"- Beauty: {bm.get('aesthetic_insight')} — elegance: {bm.get('elegance_factor')}\n")
    for tr in ta.get("truth_emergence", []) or []:
        lines.append(f"- Truth: {tr.get('truth_revealed')} — align: {tr.get('reality_alignment')}\n")
    for go in ta.get("goodness_orientation", []) or []:
        lines.append(f"- Goodness: {go.get('life_affirming_direction')} — benefit: {go.get('stakeholder_benefit')}\n")
    lines.append(_kv("coherence_quality", ta.get("coherence_quality")))
    sd = (l3 or {}).get("sovereignty_development", {})
    lines.append("\n### Sovereignty Development\n")
    for se in sd.get("sentience_expansion", []) or []:
        lines.append(f"- Sentience: {se.get('awareness_deepening')} — empathy: {se.get('empathy_development')}\n")
    for ii in sd.get("intelligence_integration", []) or []:
        lines.append(f"- Intelligence: {ii.get('sense_making_advancement')} — complexity: {ii.get('complexity_navigation')}\n")
    for am in sd.get("agency_manifestation", []) or []:
        lines.append(f"- Agency: {am.get('responsibility_taking')} — action: {am.get('purposeful_action')}\n")

    # Layer 4
    lines.append("\n## Layer 4 — Learning & Development\n")
    tll = (l4 or {}).get("triple_loop_learning", {})
    lines.append("### Triple Loop Learning\n")
    for s in tll.get("single_loop", []) or []:
        lines.append(f"- Single: {s.get('error_correction')} — {s.get('process_improvement')}\n")
    for d in tll.get("double_loop", []) or []:
        lines.append(f"- Double: {d.get('assumption_questioning')} — {d.get('mental_model_shift')}\n")
    for t in tll.get("triple_loop", []) or []:
        lines.append(f"- Triple: {t.get('context_examination')} — {t.get('paradigm_transformation')}\n")
    wdp = (l4 or {}).get("warm_data_patterns", {})
    lines.append("\n### Warm Data Patterns\n")
    for ri in wdp.get("relational_insights", []) or []:
        lines.append(f"- Relation: {', '.join(ri.get('relationship_between', []) or [])} — {ri.get('pattern')} ({ri.get('systemic_impact')})\n")
    for tc in wdp.get("transcontextual_connections", []) or []:
        lines.append(f"- Transcontext: {', '.join(tc.get('contexts', []) or [])} — {tc.get('emergent_property')}\n")
    for ls in wdp.get("living_systems_recognition", []) or []:
        lines.append(f"- Living: {ls.get('system_characteristic')} — health: {ls.get('health_indicator')}\n")
    ke = (l4 or {}).get("knowledge_evolution", {})
    lines.append("\n### Knowledge Evolution\n")
    for ic in ke.get("insights_captured", []) or []:
        lines.append(f"- Insight: {ic.get('insight')} — application: {ic.get('application_potential')} — path: {ic.get('integration_path')}\n")
    for wm in ke.get("wisdom_moments", []) or []:
        lines.append(f"- Wisdom: {wm.get('wisdom_expression')} — depth: {wm.get('depth_indicator')} — impact: {wm.get('collective_impact')}\n")
    for cb in ke.get("capacity_building", []) or []:
        lines.append(f"- Capacity: {cb.get('capacity')} — trajectory: {cb.get('development_trajectory')}\n")

    return "".join(lines)

