from typing import Dict, Any

def merge_filters(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two filter dictionaries."""
    merged = dict(base)
    merged.update(extra)
    return merged
