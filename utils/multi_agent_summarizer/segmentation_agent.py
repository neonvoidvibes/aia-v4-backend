import re
from typing import List, Dict, Any


class SegmentationAgent:
    def run(self, text: str) -> List[Dict[str, Any]]:
        # Simple heuristic:
        # 1) Try to segment by timestamps like [HH:MM:SS], HH:MM:SS, or HH:MM
        # 2) Fallback to fixed-size character windows approximating 30 minutes with 15-minute overlap
        if not text:
            return []

        ts_pattern = re.compile(r"(?:\[)?(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?(?:\])?")
        positions = [(m.start(), m.groupdict()) for m in ts_pattern.finditer(text)]

        segments: List[Dict[str, Any]] = []
        if len(positions) >= 3:  # use timestamp-based segmentation
            # Build boundaries at detected timestamps, but keep chronological order
            cut_points = [pos for (pos, _gd) in positions]
            cut_points = sorted(set([0] + cut_points + [len(text)]))
            # ensure minimum chunk size to avoid tiny segments
            min_chunk = max(2000, len(text) // 40)
            buckets = []
            for i in range(len(cut_points) - 1):
                a, b = cut_points[i], cut_points[i + 1]
                if b - a >= min_chunk:
                    buckets.append((a, b))
            if not buckets:
                buckets = [(0, len(text))]

            # map to minutes by naive diff between timestamp groups if possible
            def parse_min(d):
                try:
                    h = int(d.get("h") or 0)
                    m = int(d.get("m") or 0)
                    s = int(d.get("s") or 0)
                    return h * 60 + m + (s / 60.0)
                except Exception:
                    return None

            mins = [parse_min(gd) for (_p, gd) in positions]
            approx_mins = [m for m in mins if m is not None]
            # create segments ~30 minutes, but it's okay if irregular based on timestamps
            for i, (a, b) in enumerate(buckets, start=1):
                start_min = approx_mins[i - 1] if i - 1 < len(approx_mins) else (i - 1) * 30
                end_min = approx_mins[i] if i < len(approx_mins) else start_min + 30
                seg_text = text[a:b]
                segments.append({
                    "id": f"seg:{i}",
                    "start_min": max(0, int(start_min)),
                    "end_min": max(int(start_min) + 1, int(end_min)),
                    "text": seg_text.strip(),
                    "bridge_in": "",
                    "bridge_out": "",
                })
        else:
            # Fixed-size windows: 30-min segment approximated by ~10k chars with 5k overlap
            seg_chars = 10000
            overlap = 5000
            i = 0
            seg_idx = 1
            approx_min_per_char = 30.0 / max(seg_chars, 1)
            while i < len(text):
                a = i
                b = min(len(text), i + seg_chars)
                seg_text = text[a:b]
                start_min = int(a * approx_min_per_char)
                end_min = int(start_min + 30)
                segments.append({
                    "id": f"seg:{seg_idx}",
                    "start_min": start_min,
                    "end_min": end_min,
                    "text": seg_text.strip(),
                    "bridge_in": "",
                    "bridge_out": "",
                })
                seg_idx += 1
                if b == len(text):
                    break
                i = max(a + seg_chars - overlap, a + 1)

        return segments

