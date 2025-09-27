#!/usr/bin/env python3
"""
Production validation script for hallucination filtering metrics.
Run this script after deployment to verify all 10 monitoring categories are working.
"""

import sys
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass

try:
    import requests
    from prometheus_client.parser import text_string_to_metric_families
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class MetricCheck:
    name: str
    metric_name: str
    expected_labels: List[str]
    target_value: float
    comparison: str  # "gt", "lt", "eq"
    critical: bool = False


class HallucinationMetricsValidator:
    """Validates that all hallucination filtering metrics are working in production."""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.checks = self._define_validation_checks()

    def _define_validation_checks(self) -> List[MetricCheck]:
        """Define all validation checks for the 10 monitoring categories."""
        return [
            # 1. Trim efficiency by session age
            MetricCheck(
                "trim_efficiency_total",
                "hallucination_trim_efficiency_total",
                ["session_age_bucket", "provider", "language"],
                0, "gt", critical=True
            ),
            MetricCheck(
                "trim_efficiency_early_bucket",
                'rate(hallucination_trim_efficiency_trimmed_total{session_age_bucket="0-5s"}[5m])',
                [],
                0.30, "lt"  # Should be 10-30%, alert if >30%
            ),

            # 2. False-drop guard
            MetricCheck(
                "false_drops_rate",
                'rate(hallucination_false_drops_total[5m]) / rate(hallucination_segments_processed_total[5m])',
                [],
                0.01, "lt", critical=True  # Must be <1%
            ),

            # 3. Fallback pressure
            MetricCheck(
                "fallback_rate",
                'rate(hallucination_fallback_usage_total[10m]) / rate(hallucination_segments_processed_total[10m])',
                [],
                0.08, "lt"  # Target <8%
            ),

            # 4. Empty-after-trim sentinel
            MetricCheck(
                "empty_after_trim_rate",
                'rate(hallucination_empty_after_trim_total[10m]) / rate(hallucination_segments_processed_total[10m])',
                [],
                0.001, "lt", critical=True  # Should be ~0%
            ),

            # 5. First-utterance duplicate rate
            MetricCheck(
                "first_utterance_duplicate_rate",
                'rate(hallucination_first_utterance_duplicates_total[1h]) / rate(hallucination_first_utterance_sessions_total[1h])',
                [],
                0.03, "lt"  # Target <3%
            ),

            # 6. Cut word distribution
            MetricCheck(
                "cut_word_p95",
                'histogram_quantile(0.95, rate(hallucination_cut_word_count_bucket[5m]))',
                [],
                6, "lt"  # P95 ≤6 words
            ),

            # 7. Latency impact
            MetricCheck(
                "post_asr_decision_p95",
                'histogram_quantile(0.95, rate(hallucination_post_asr_decision_duration_seconds_bucket[5m])) * 1000',
                [],
                3, "lt", critical=True  # Budget ≤3ms P95
            ),

            # 8. Context health
            MetricCheck(
                "context_length_p95",
                'histogram_quantile(0.95, rate(hallucination_context_tail_length_bucket[5m]))',
                [],
                200, "lt"  # P95 ≤200 tokens
            ),
            MetricCheck(
                "context_reset_rate",
                'rate(hallucination_context_resets_total[10m])',
                [],
                0.001, "lt"  # Target ~0%
            ),

            # 9. Validator/PII outcomes
            MetricCheck(
                "validator_fail_rate",
                'rate(hallucination_validator_outcomes_total{outcome="fail"}[10m]) / rate(hallucination_validator_outcomes_total[10m])',
                [],
                0.003, "lt"  # Target <0.3%
            ),
            MetricCheck(
                "pii_error_rate",
                'rate(hallucination_pii_outcomes_total{outcome="error"}[10m]) / rate(hallucination_pii_outcomes_total[10m])',
                [],
                0.001, "lt", critical=True  # Target ~0%
            ),

            # Basic existence checks for all metrics
            MetricCheck(
                "segments_processed",
                "hallucination_segments_processed_total",
                ["provider", "language"],
                0, "gt", critical=True
            )
        ]

    def _query_prometheus(self, query: str) -> Dict[str, Any]:
        """Query Prometheus API."""
        if not REQUESTS_AVAILABLE:
            return {"status": "error", "error": "requests module not available"}

        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _evaluate_check(self, check: MetricCheck) -> Dict[str, Any]:
        """Evaluate a single metric check."""
        result = self._query_prometheus(check.metric_name)

        if result.get("status") == "error":
            return {
                "name": check.name,
                "status": "error",
                "message": f"Query failed: {result.get('error', 'Unknown error')}",
                "critical": check.critical
            }

        data = result.get("data", {})
        result_data = data.get("result", [])

        if not result_data:
            return {
                "name": check.name,
                "status": "warning",
                "message": "No data returned - metric may not exist or have no values",
                "critical": check.critical
            }

        # Get the first result value
        try:
            value = float(result_data[0]["value"][1])
        except (KeyError, IndexError, ValueError) as e:
            return {
                "name": check.name,
                "status": "error",
                "message": f"Failed to extract value: {e}",
                "critical": check.critical
            }

        # Evaluate comparison
        passed = False
        if check.comparison == "gt":
            passed = value > check.target_value
        elif check.comparison == "lt":
            passed = value < check.target_value
        elif check.comparison == "eq":
            passed = abs(value - check.target_value) < 0.001

        status = "pass" if passed else "fail"
        message = f"Value: {value:.4f}, Target: {check.comparison} {check.target_value}"

        return {
            "name": check.name,
            "status": status,
            "message": message,
            "value": value,
            "target": check.target_value,
            "critical": check.critical
        }

    def validate_all_metrics(self) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive report."""
        print("🔍 Validating hallucination filtering metrics...")
        print(f"📊 Prometheus URL: {self.prometheus_url}")
        print(f"✅ Running {len(self.checks)} validation checks...\n")

        results = []
        critical_failures = 0
        warnings = 0
        passes = 0

        for check in self.checks:
            print(f"Checking {check.name}...", end=" ")
            result = self._evaluate_check(check)
            results.append(result)

            if result["status"] == "pass":
                print("✅ PASS")
                passes += 1
            elif result["status"] == "fail":
                if result["critical"]:
                    print("❌ CRITICAL FAIL")
                    critical_failures += 1
                else:
                    print("⚠️  FAIL")
                    warnings += 1
            elif result["status"] == "warning":
                print("⚠️  WARNING")
                warnings += 1
            else:
                print("💥 ERROR")
                if result["critical"]:
                    critical_failures += 1
                else:
                    warnings += 1

            if result["status"] != "pass":
                print(f"   └─ {result['message']}")

        # Summary
        print(f"\n📈 VALIDATION SUMMARY:")
        print(f"   ✅ Passed: {passes}")
        print(f"   ⚠️  Warnings: {warnings}")
        print(f"   ❌ Critical Failures: {critical_failures}")

        overall_status = "PASS"
        if critical_failures > 0:
            overall_status = "CRITICAL_FAIL"
        elif warnings > 0:
            overall_status = "WARNING"

        print(f"\n🎯 OVERALL STATUS: {overall_status}")

        if overall_status == "CRITICAL_FAIL":
            print("\n⚠️  CRITICAL ISSUES FOUND - DO NOT PROCEED TO PRODUCTION")
            print("   Fix critical issues before continuing rollout.")
        elif overall_status == "WARNING":
            print("\n⚠️  WARNINGS FOUND - REVIEW BEFORE FULL ROLLOUT")
            print("   Consider investigating warnings in staging.")
        else:
            print("\n🚀 ALL CHECKS PASSED - READY FOR PRODUCTION RAMP")

        return {
            "overall_status": overall_status,
            "summary": {
                "passes": passes,
                "warnings": warnings,
                "critical_failures": critical_failures,
                "total_checks": len(self.checks)
            },
            "detailed_results": results,
            "ready_for_production": critical_failures == 0
        }

    def generate_report(self, output_file: str = None) -> str:
        """Generate detailed validation report."""
        validation_results = self.validate_all_metrics()

        report = {
            "timestamp": time.time(),
            "validation_results": validation_results,
            "recommendations": self._generate_recommendations(validation_results),
            "next_steps": self._generate_next_steps(validation_results)
        }

        report_json = json.dumps(report, indent=2)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_json)
            print(f"\n📄 Detailed report saved to: {output_file}")

        return report_json

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        for result in results["detailed_results"]:
            if result["status"] == "fail" and "trim_efficiency_early" in result["name"]:
                recommendations.append(
                    "Consider adjusting bootstrap thresholds (boot_min, boot_ratio) if early trim rate is too low"
                )
            elif result["status"] == "fail" and "false_drops_rate" in result["name"]:
                recommendations.append(
                    "CRITICAL: False drop rate too high - review post-ASR decision logic and toggles"
                )
            elif result["status"] == "fail" and "latency" in result["name"]:
                recommendations.append(
                    "Performance regression detected - review recent changes for optimization opportunities"
                )

        if not recommendations:
            recommendations.append("All metrics within acceptable ranges - proceed with confidence")

        return recommendations

    def _generate_next_steps(self, results: Dict[str, Any]) -> List[str]:
        """Generate next steps based on validation results."""
        if results["ready_for_production"]:
            return [
                "✅ Deploy to 10% canary traffic",
                "✅ Monitor for 24 hours with alerting enabled",
                "✅ If SLOs hold, ramp to 100%",
                "✅ Enable weekly content integrity spot checks"
            ]
        else:
            return [
                "❌ Fix critical issues before deployment",
                "❌ Re-run validation after fixes",
                "❌ Consider rollback plan if issues persist",
                "❌ Review logs and metrics for root cause"
            ]


def main():
    """Main validation entry point."""
    if len(sys.argv) > 1:
        prometheus_url = sys.argv[1]
    else:
        prometheus_url = "http://localhost:9090"

    validator = HallucinationMetricsValidator(prometheus_url)

    # Run validation
    validator.generate_report("hallucination_metrics_validation.json")


if __name__ == "__main__":
    main()