"""
Metrics tracking for ASR hallucination detection and transcription service.
"""
import logging
from collections import defaultdict
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Simple in-memory counters for now (can be extended to Prometheus/DataDog later)
_counters: Dict[str, int] = defaultdict(int)
_histograms: Dict[str, list] = defaultdict(list)

class MetricCounter:
    def __init__(self, name: str, description: str, label_names: list):
        self.name = name
        self.description = description
        self.label_names = label_names or []

    def labels(self, **kwargs):
        """Return a labeled counter instance"""
        # Allow partial labels, missing values default to "unknown"
        label_values = {
            key: kwargs.get(key, "unknown")
            for key in self.label_names
        }
        # Include any ad-hoc labels that were passed explicitly
        for key, value in kwargs.items():
            if key not in label_values:
                label_values[key] = value
        return LabeledCounter(self.name, label_values)

class LabeledCounter:
    def __init__(self, name: str, label_values: dict):
        self.name = name
        self.label_values = label_values

    def inc(self, value: int = 1):
        """Increment the counter"""
        key = f"{self.name}_{self._make_key()}"
        if key not in _counters:
            _counters[key] = 0
        _counters[key] += value
        logger.debug(f"Counter {key} incremented by {value}, total: {_counters[key]}")

    def _make_key(self) -> str:
        return "_".join(f"{k}_{v}" for k, v in sorted(self.label_values.items()))

class Histogram:
    def __init__(self, name: str, description: str, label_names: list, buckets: list):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets
        _histograms[name] = []

    def labels(self, **kwargs):
        """Return a labeled histogram instance"""
        label_values = {
            key: kwargs.get(key, "unknown")
            for key in self.label_names
        }
        for key, value in kwargs.items():
            if key not in label_values:
                label_values[key] = value
        return LabeledHistogram(self.name, label_values)

class LabeledHistogram:
    def __init__(self, name: str, label_values: dict):
        self.name = name
        self.label_values = label_values

    def observe(self, value: float):
        """Observe a value"""
        key = f"{self.name}_{self._make_key()}"
        if key not in _histograms:
            _histograms[key] = []
        _histograms[key].append(value)
        logger.debug(f"Histogram {key} observed value: {value}")

    def _make_key(self) -> str:
        return "_".join(f"{k}_{v}" for k, v in sorted(self.label_values.items()))

# Initialize metrics
HALLU_TRIMS = MetricCounter("asr_hallu_trims_total", "Count of overlap trims", ["reason", "provider"])
HALLU_DECISIONS = Histogram("asr_hallu_decision", "Detector signals", ["metric"], buckets=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])

PIPELINE_FALLBACK_USED = MetricCounter(
    "transcript_pipeline_fallback_total",
    "Segments that fell back to original ASR text",
    ["reason"]
)

PIPELINE_DROPS = MetricCounter(
    "transcript_pipeline_drop_total",
    "Segments dropped after post-ASR pipeline",
    ["reason"]
)

PIPELINE_PII_DECISIONS = MetricCounter(
    "transcript_pipeline_pii_total",
    "PII filter decisions",
    ["result"]
)

PIPELINE_ZERO_AFTER_DEDUP = MetricCounter(
    "transcript_pipeline_zero_after_dedup_total",
    "Segments that were empty after dedup/trim",
    ["source"]
)

PIPELINE_ASR_EMPTY = MetricCounter(
    "transcript_pipeline_asr_empty_total",
    "Segments where ASR produced no text",
    []
)

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of all metrics"""
    return {
        "counters": dict(_counters),
        "histograms": {k: {"count": len(v), "sum": sum(v), "avg": sum(v)/len(v) if v else 0} for k, v in _histograms.items()}
    }
