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
    def __init__(self, name: str, description: str, labels: list):
        self.name = name
        self.description = description
        self.labels = labels

    def labels(self, **kwargs):
        """Return a labeled counter instance"""
        return LabeledCounter(self.name, kwargs)

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
    def __init__(self, name: str, description: str, labels: list, buckets: list):
        self.name = name
        self.description = description
        self.labels = labels
        self.buckets = buckets
        _histograms[name] = []

    def labels(self, **kwargs):
        """Return a labeled histogram instance"""
        return LabeledHistogram(self.name, kwargs)

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
HALLU_DECISIONS = Histogram("asr_hallu_decision", "Detector signals", ["metric"], buckets=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of all metrics"""
    return {
        "counters": dict(_counters),
        "histograms": {k: {"count": len(v), "sum": sum(v), "avg": sum(v)/len(v) if v else 0} for k, v in _histograms.items()}
    }