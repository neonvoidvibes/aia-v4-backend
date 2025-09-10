L1_HEADER = "# Layer 1 — Traditional Business"
L2_HEADER = "# Layer 2 — Collective Intelligence"
L3_HEADER = "# Layer 3 — Wisdom Integration"
L4_HEADER = "# Layer 4 — Learning & Development"

CONF_HEADER = "## Confidence"

def header(title: str) -> str:
    return f"{title}\n"

def kv_line(pairs: dict) -> str:
    parts = []
    for k, v in pairs.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            parts.append(f"{k}: {', '.join(str(x) for x in v)}")
        else:
            parts.append(f"{k}: {v}")
    return "- " + " | ".join(parts) + "\n"

def wrap_section(title: str, lines: list[str]) -> str:
    out = [f"{title}\n"]
    out.extend(lines)
    out.append("\n")
    return "".join(out)

def make_confidence_md(conf: dict[str, float]) -> str:
    lines = []
    for k in ["layer1","layer2","layer3","layer4"]:
        v = conf.get(k, 0.0)
        lines.append(f"- {k}: {v:.2f}\n")
    return wrap_section(CONF_HEADER, lines)

