import re
from typing import Dict, Any, List

_LIST_KEYS = {
    'segment_ids', 'dependencies', 'evidence_ids', 'predicted_outcomes',
    'relationship_between', 'contexts', 'energy_shifts', 'required_participants'
}

def _clean_scalar(v: str) -> str:
    if not isinstance(v, str):
        return v
    # Strip markdown emphasis and verbatim wrappers
    v = v.strip()
    v = v.replace('**', '')
    v = v.replace('<verbatim>', '').replace('</verbatim>', '')
    return v


def _parse_fields(line: str) -> Dict[str, Any]:
    # expects '- key: val | key2: val2 | list: a, b'
    if line.startswith('- '):
        line = line[2:]
    parts = [p.strip() for p in line.split('|')]
    out: Dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        if ':' not in part:
            continue
        k, v = part.split(':', 1)
        k = k.strip()
        v = _clean_scalar(v.strip())
        # Do not auto-listize here; handled downstream for known keys
        out[k] = v
    return out


def parse_full_markdown(md: str) -> Dict[str, Any]:
    # Very lightweight parser mapping expected sections to schema-ish dict
    # Unknown fields are ignored. Missing fields are defaulted later by Integration.
    lines = md.splitlines()
    res: Dict[str, Any] = {
        "layer1": {"meeting_metadata": {}, "actionable_outputs": {"decisions": [], "tasks": [], "next_meetings": []}},
        "layer2": {"mirror_level": {"explicit_themes": [], "stated_agreements": [], "direct_quotes": [], "participation_patterns": {"energy_shifts": [], "engagement_distribution": "unknown"}},
                   "lens_level": {"hidden_patterns": [], "unspoken_tensions": [], "group_dynamics": {"emotional_undercurrents": "", "power_dynamics": ""}, "paradoxes_contradictions": []},
                   "portal_level": {"emergent_possibilities": [], "intervention_opportunities": [], "paradigm_shifts": []}},
        "layer3": {"connectedness_patterns": {"self_awareness": [], "interpersonal_dynamics": [], "systemic_understanding": []},
                   "transcendental_alignment": {"beauty_moments": [], "truth_emergence": [], "goodness_orientation": [], "coherence_quality": ""},
                   "sovereignty_development": {"sentience_expansion": [], "intelligence_integration": [], "agency_manifestation": []}},
        "layer4": {"triple_loop_learning": {"single_loop": [], "double_loop": [], "triple_loop": []},
                   "warm_data_patterns": {"relational_insights": [], "transcontextual_connections": [], "living_systems_recognition": []},
                   "knowledge_evolution": {"insights_captured": [], "wisdom_moments": [], "capacity_building": []}},
        "confidence": {}
    }

    sec = None
    sub = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # headers
        if line.startswith('# Layer 1'):
            sec = 'layer1'; sub = None; continue
        if line.startswith('# Layer 2'):
            sec = 'layer2'; sub = None; continue
        if line.startswith('# Layer 3'):
            sec = 'layer3'; sub = None; continue
        if line.startswith('# Layer 4'):
            sec = 'layer4'; sub = None; continue
        if line.startswith('## Confidence'):
            sec = 'confidence'; sub = None; continue
        # subsections
        if line.startswith('### '):
            sub = line[4:].strip().lower()
            continue

        if line.startswith('- '):
            fields = _parse_fields(line)
            # Normalize known list fields to lists
            for lk in _LIST_KEYS:
                if lk in fields:
                    val = fields[lk]
                    if isinstance(val, str):
                        # split by comma
                        fields[lk] = [ _clean_scalar(x.strip()) for x in val.split(',') if x.strip() ]
                    elif isinstance(val, list):
                        fields[lk] = [_clean_scalar(x) for x in val]
                    else:
                        fields[lk] = [ _clean_scalar(val) ]
            if sec == 'confidence':
                # '- layer1: 0.75'
                for k, v in fields.items():
                    try:
                        res['confidence'][k] = float(v)
                    except Exception:
                        pass
                continue

            if sec == 'layer1':
                if sub is None:
                    # meeting metadata kv lines
                    res['layer1']['meeting_metadata'].update(fields)
                elif sub == 'decisions':
                    res['layer1']['actionable_outputs']['decisions'].append(fields)
                elif sub == 'tasks':
                    res['layer1']['actionable_outputs']['tasks'].append(fields)
                elif sub == 'next meetings':
                    res['layer1']['actionable_outputs']['next_meetings'].append(fields)

            elif sec == 'layer2':
                if sub is None or 'mirror' in (sub or ''):
                    # Mirror bullets
                    # detect type by available keys
                    if 'theme' in fields:
                        # ensure required fields
                        fields.setdefault('context', '')
                        res['layer2']['mirror_level']['explicit_themes'].append(fields)
                    elif 'agreement' in fields:
                        fields.setdefault('segment_ids', [])
                        res['layer2']['mirror_level']['stated_agreements'].append(fields)
                    elif 'quote' in fields:
                        fields.setdefault('context', '')
                        fields.setdefault('significance', '')
                        fields.setdefault('segment_ids', [])
                        res['layer2']['mirror_level']['direct_quotes'].append(fields)
                    elif 'engagement_distribution' in fields or 'energy_shifts' in fields:
                        res['layer2']['mirror_level']['participation_patterns'].update(fields)
                elif 'lens' in sub:
                    if 'pattern' in fields:
                        res['layer2']['lens_level']['hidden_patterns'].append(fields)
                    elif 'tension' in fields:
                        res['layer2']['lens_level']['unspoken_tensions'].append(fields)
                    elif 'emotional_undercurrents' in fields or 'power_dynamics' in fields:
                        res['layer2']['lens_level']['group_dynamics'].update(fields)
                    elif 'paradox' in fields:
                        res['layer2']['lens_level']['paradoxes_contradictions'].append(fields)
                elif 'portal' in sub:
                    if 'possibility' in fields:
                        # ensure grounding_in_lens/evidence_ids types
                        if 'evidence_ids' in fields and not isinstance(fields['evidence_ids'], list):
                            fields['evidence_ids'] = [fields['evidence_ids']]
                        if 'grounding_in_lens' not in fields and fields.get('evidence_ids'):
                            lens_eids = [e for e in fields['evidence_ids'] if isinstance(e, str) and e.startswith('lens:')]
                            if lens_eids:
                                fields['grounding_in_lens'] = lens_eids[0]
                        res['layer2']['portal_level']['emergent_possibilities'].append(fields)
                    elif 'leverage_point' in fields:
                        if 'evidence_ids' in fields and not isinstance(fields['evidence_ids'], list):
                            fields['evidence_ids'] = [fields['evidence_ids']]
                        # normalize predicted_outcomes
                        if 'predicted_outcomes' in fields and not isinstance(fields['predicted_outcomes'], list):
                            fields['predicted_outcomes'] = [fields['predicted_outcomes']]
                        res['layer2']['portal_level']['intervention_opportunities'].append(fields)
                    elif 'shift' in fields:
                        if 'evidence_ids' in fields and not isinstance(fields['evidence_ids'], list):
                            fields['evidence_ids'] = [fields['evidence_ids']]
                        res['layer2']['portal_level']['paradigm_shifts'].append(fields)

            elif sec == 'layer3':
                if 'insight' in fields:
                    res['layer3']['connectedness_patterns']['self_awareness'].append(fields)
                elif 'relationship_shift' in fields:
                    res['layer3']['connectedness_patterns']['interpersonal_dynamics'].append(fields)
                elif 'systems_insight' in fields:
                    res['layer3']['connectedness_patterns']['systemic_understanding'].append(fields)
                elif 'aesthetic_insight' in fields:
                    res['layer3']['transcendental_alignment'].setdefault('beauty_moments', []).append(fields)
                elif 'truth_revealed' in fields:
                    res['layer3']['transcendental_alignment'].setdefault('truth_emergence', []).append(fields)
                elif 'life_affirming_direction' in fields:
                    res['layer3']['transcendental_alignment'].setdefault('goodness_orientation', []).append(fields)
                elif 'coherence_quality' in fields:
                    res['layer3']['transcendental_alignment'].update(fields)
                elif 'responsibility_taking' in fields:
                    res['layer3']['sovereignty_development'].setdefault('agency_manifestation', []).append(fields)
                elif 'sense_making_advancement' in fields:
                    res['layer3']['sovereignty_development'].setdefault('intelligence_integration', []).append(fields)
                elif 'awareness_deepening' in fields:
                    res['layer3']['sovereignty_development'].setdefault('sentience_expansion', []).append(fields)

            elif sec == 'layer4':
                if 'error_correction' in fields:
                    res['layer4']['triple_loop_learning']['single_loop'].append(fields)
                elif 'assumption_questioning' in fields:
                    res['layer4']['triple_loop_learning']['double_loop'].append(fields)
                elif 'context_examination' in fields:
                    res['layer4']['triple_loop_learning']['triple_loop'].append(fields)
                elif 'relationship_between' in fields:
                    res['layer4']['warm_data_patterns']['relational_insights'].append(fields)
                elif 'contexts' in fields:
                    res['layer4']['warm_data_patterns']['transcontextual_connections'].append(fields)
                elif 'system_characteristic' in fields:
                    res['layer4']['warm_data_patterns']['living_systems_recognition'].append(fields)
                elif 'insight' in fields and 'integration_path' in fields:
                    res['layer4']['knowledge_evolution']['insights_captured'].append(fields)
                elif 'wisdom_expression' in fields:
                    res['layer4']['knowledge_evolution']['wisdom_moments'].append(fields)
                elif 'capacity' in fields:
                    res['layer4']['knowledge_evolution']['capacity_building'].append(fields)

    # Coerce common numeric fields & defaults
    try:
        mm = res.get('layer1', {}).get('meeting_metadata', {})
        # duration
        if isinstance(mm.get('duration_minutes'), str):
            mm['duration_minutes'] = int(mm['duration_minutes']) if mm['duration_minutes'].isdigit() else 0
        elif not isinstance(mm.get('duration_minutes'), int):
            mm['duration_minutes'] = 0
        # participants
        if isinstance(mm.get('participants_count'), str):
            mm['participants_count'] = int(mm['participants_count']) if mm['participants_count'].isdigit() else 1
        elif not isinstance(mm.get('participants_count'), int):
            mm['participants_count'] = 1
    except Exception:
        pass

    # Coerce numbers within structures
    for item in res.get('layer2', {}).get('mirror_level', {}).get('explicit_themes', []):
        try:
            if isinstance(item.get('frequency'), str) and item['frequency'].isdigit():
                item['frequency'] = int(item['frequency'])
        except Exception:
            pass

    for item in res.get('layer2', {}).get('portal_level', {}).get('intervention_opportunities', []):
        try:
            ps = item.get('probability_score')
            if isinstance(ps, str):
                ps = ps.replace('%','').strip()
                item['probability_score'] = float(ps)/100.0 if '%' in item.get('probability_score','') else float(ps)
        except Exception:
            pass

    # Ensure required fields exist where schemas expect them
    for item in res.get('layer2', {}).get('mirror_level', {}).get('explicit_themes', []):
        item.setdefault('context', '')
        if isinstance(item.get('segment_ids'), str):
            item['segment_ids'] = [item['segment_ids']]
        item.setdefault('segment_ids', [])
    for item in res.get('layer2', {}).get('mirror_level', {}).get('stated_agreements', []):
        if isinstance(item.get('segment_ids'), str):
            item['segment_ids'] = [item['segment_ids']]
        item.setdefault('segment_ids', [])
    for item in res.get('layer2', {}).get('mirror_level', {}).get('direct_quotes', []):
        item.setdefault('context', '')
        item.setdefault('significance', '')
        if isinstance(item.get('segment_ids'), str):
            item['segment_ids'] = [item['segment_ids']]
        item.setdefault('segment_ids', [])
    for item in res.get('layer1', {}).get('actionable_outputs', {}).get('tasks', []):
        if isinstance(item.get('dependencies'), str):
            item['dependencies'] = [item['dependencies']]
        item.setdefault('dependencies', [])
    for cat in ['emergent_possibilities','intervention_opportunities','paradigm_shifts']:
        for item in res.get('layer2', {}).get('portal_level', {}).get(cat, []):
            if isinstance(item.get('evidence_ids'), str):
                item['evidence_ids'] = [item['evidence_ids']]
            item.setdefault('evidence_ids', [])
            if cat == 'emergent_possibilities' and 'grounding_in_lens' not in item and item.get('evidence_ids'):
                lens_eids = [e for e in item['evidence_ids'] if isinstance(e, str) and e.startswith('lens:')]
                if lens_eids:
                    item['grounding_in_lens'] = lens_eids[0]
            if cat == 'paradigm_shifts':
                item.setdefault('indicators', '')
                item.setdefault('readiness_assessment', '')

    # Ensure next_meetings.required_participants is a list
    for nm in res.get('layer1', {}).get('actionable_outputs', {}).get('next_meetings', []):
        if isinstance(nm.get('required_participants'), str):
            nm['required_participants'] = [nm['required_participants']]
        nm.setdefault('required_participants', [])
    # Layer3 sovereignty_development: ensure required subfields
    for item in res.get('layer3', {}).get('sovereignty_development', {}).get('intelligence_integration', []):
        item.setdefault('complexity_navigation', '')
    # Layer4 triple_loop: ensure required subfields
    for item in res.get('layer4', {}).get('triple_loop_learning', {}).get('single_loop', []):
        item.setdefault('process_improvement', '')
    for item in res.get('layer4', {}).get('triple_loop_learning', {}).get('double_loop', []):
        item.setdefault('mental_model_shift', '')
    for item in res.get('layer4', {}).get('triple_loop_learning', {}).get('triple_loop', []):
        item.setdefault('paradigm_transformation', '')
    return res
    
