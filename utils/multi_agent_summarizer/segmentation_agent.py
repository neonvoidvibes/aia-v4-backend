import re
from typing import List, Dict, Any

def _parse_timestamps(text: str) -> List[Dict[str, Any]]:
    # Matches [HH:MM:SS] or HH:MM:SS or HH:MM
    pat = re.compile(r"\[?(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?\]?")
    out = []
    for m in pat.finditer(text):
        try:
            h = int(m.group('h'))
            mi = int(m.group('m'))
            s = int(m.group('s') or 0)
            minute = h * 60 + mi + s / 60.0
            out.append({"pos": m.start(), "minute": minute})
        except Exception:
            continue
    return out


class SegmentationAgent:
    def run(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        segments: List[Dict[str, Any]] = []
        stamps = _parse_timestamps(text)
        if stamps:
            # Build time-windowed segments around minutes (30m windows with 15m overlap)
            minutes = [s["minute"] for s in stamps]
            t_start = int(minutes[0])
            t_end = int(minutes[-1])
            if t_end <= t_start:
                t_end = t_start + 60
            window = 30
            overlap = 15
            seg_idx = 1
            t = t_start
            while t < t_end:
                w_start = t
                w_end = min(t_end, t + window)
                # Find character bounds spanning this time window
                idxs = [s for s in stamps if w_start <= s["minute"] <= w_end]
                if not idxs:
                    # include span between nearest timestamps
                    idxs = [min(stamps, key=lambda s: abs(s["minute"] - w_start)), min(stamps, key=lambda s: abs(s["minute"] - w_end))]
                    idxs = sorted(idxs, key=lambda s: s["pos"])
                a = idxs[0]["pos"]
                b = idxs[-1]["pos"]
                if a >= b:
                    b = min(len(text), a + 8000)
                seg_text = text[a:b].strip()
                if seg_text:
                    segments.append({
                        "id": f"seg:{seg_idx}",
                        "start_min": int(w_start),
                        "end_min": max(int(w_start + 1), int(w_end)),
                        "text": seg_text,
                        "bridge_in": "",
                        "bridge_out": "",
                    })
                    seg_idx += 1
                t += (window - overlap)
        else:
            # Token-ish char windows (approx) with overlap
            seg_chars = 9000
            overlap = 4500
            i = 0
            seg_idx = 1
            approx_min_per_char = 30.0 / max(seg_chars, 1)
            while i < len(text):
                a = i
                b = min(len(text), i + seg_chars)
                seg_text = text[a:b].strip()
                start_min = int(a * approx_min_per_char)
                end_min = int(start_min + 30)
                if seg_text:
                    segments.append({
                        "id": f"seg:{seg_idx}",
                        "start_min": start_min,
                        "end_min": end_min,
                        "text": seg_text,
                        "bridge_in": "",
                        "bridge_out": "",
                    })
                    seg_idx += 1
                if b == len(text):
                    break
                i = max(a + seg_chars - overlap, a + 1)

        return segments
