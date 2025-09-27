# Hallucination Filtering System Monitoring

Comprehensive monitoring and alerting for the refactored hallucination filtering system. This implementation provides 10 categories of monitoring with strict SLOs and automated validation.

## 📊 Monitoring Categories

### 1. Trim Efficiency by Session Age
**Purpose**: Validate bootstrap thresholds are working correctly
**Metrics**: `hallucination_trim_efficiency_*`
**Targets**:
- 0-5s bucket: 10-30% trim rate
- 5-30s bucket: 5-15% trim rate
- 30s+ bucket: <5% trim rate

### 2. False-Drop Guard
**Purpose**: Prevent over-aggressive filtering
**Metrics**: `hallucination_false_drops_total`
**Targets**:
- <0.5% baseline
- Alert on 1%+ spike (critical)

### 3. Fallback Pressure
**Purpose**: Monitor system stress and quality degradation
**Metrics**: `hallucination_fallback_usage_total`
**Targets**:
- 2-5% steady baseline
- Investigate drift >8%

### 4. Empty-After-Trim Sentinel
**Purpose**: Bug detection for trimming logic
**Metrics**: `hallucination_empty_after_trim_total`
**Targets**:
- ~0% (any spike indicates bug)

### 5. First-Utterance Duplicate Rate
**Purpose**: Validate bootstrap fix for greeting repeats
**Metrics**: `hallucination_first_utterance_duplicates_total`
**Targets**:
- <3% after bootstrap changes

### 6. Kept-vs-Cut Distribution
**Purpose**: Understand trimming patterns and provider differences
**Metrics**: `hallucination_cut_word_count`
**Targets**:
- P50: 0-2 words
- P95: ≤6 words

### 7. Latency Impact
**Purpose**: Ensure real-time performance
**Metrics**: `hallucination_post_asr_decision_duration_seconds`
**Targets**:
- P95 ≤3ms for post-ASR decisions
- No regression >5% end-to-end

### 8. Context Health
**Purpose**: Monitor memory usage and session stability
**Metrics**: `hallucination_context_tail_length`, `hallucination_context_resets_total`
**Targets**:
- P95 ≤200 tokens (cap enforced)
- Context resets ~0%

### 9. Validator/PII Outcomes
**Purpose**: Track integration health
**Metrics**: `hallucination_validator_outcomes_total`, `hallucination_pii_outcomes_total`
**Targets**:
- Validator fail <0.3%
- PII error ~0%

### 10. Content Integrity Spot Check
**Purpose**: Human validation of filtering quality
**Process**: Weekly manual audit
**Targets**:
- Precision/recall ≥0.95 on trim decisions

## 🚨 Alert Configuration

### Critical Alerts
- **False drops >1%**: Immediate investigation required
- **Empty after trim**: Bug in trimming logic
- **PII errors >0.1%**: Integration failure
- **Latency >3ms P95**: Performance degradation

### Warning Alerts
- **Fallback pressure >8%**: Quality investigation
- **Bootstrap efficiency low**: Threshold tuning needed
- **Validator fails >0.3%**: Text quality issues

## 📈 Dashboard

The monitoring dashboard includes:
- Real-time safety overview
- Trim efficiency by session age
- Fallback pressure trends
- Performance metrics (P95 latency)
- Context health indicators
- Validator/PII outcome rates

Location: `config/hallucination_dashboard.json`

## 🔧 Setup Instructions

### 1. Deploy Metrics Collection
```bash
# Ensure Prometheus client is available
pip install prometheus-client

# Metrics are automatically collected when the system runs
# Check utils/hallucination_metrics.py for integration
```

### 2. Configure Alerts
```bash
# Deploy alert rules to Prometheus/Alertmanager
kubectl apply -f config/hallucination_alerts.yml

# Or for local Prometheus:
cp config/hallucination_alerts.yml /etc/prometheus/rules/
```

### 3. Setup Dashboard
```bash
# Import dashboard to Grafana
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @config/hallucination_dashboard.json
```

### 4. Validation Script
```bash
# Run validation against Prometheus
python scripts/validate_hallucination_metrics.py http://prometheus:9090

# This will generate a comprehensive report
```

## 🚀 Production Rollout Plan

### Phase 1: Staging Validation (1-2 days)
1. Deploy to staging with all metrics enabled
2. Run validation script every hour
3. Monitor all 10 categories for baseline establishment
4. Verify alerts trigger correctly with synthetic failures

### Phase 2: Canary Deployment (24-48 hours)
1. Deploy to 10% of production traffic
2. Monitor SLOs continuously:
   ```bash
   # Automated checks every 15 minutes
   */15 * * * * python scripts/validate_hallucination_metrics.py
   ```
3. Alert thresholds:
   - False drops >1% → immediate rollback
   - Empty after trim any → immediate investigation
   - Latency regression >5% → performance review

### Phase 3: Full Rollout (if SLOs hold)
1. Ramp to 100% over 4-6 hours
2. Continue monitoring for 1 week
3. Weekly content integrity audits begin
4. Baseline adjustment if needed

### Rollback Triggers
- False drop rate >1% sustained for >5 minutes
- Empty after trim incidents >0
- PII error rate >0.5%
- Performance regression >10%
- Manual content audit precision <0.90

## 📊 SLO Definitions

| Metric | Target | Warning | Critical |
|--------|---------|---------|----------|
| False Drop Rate | <0.5% | >0.5% | >1.0% |
| Empty After Trim | 0% | Any | Any |
| Fallback Rate | 2-5% | >5% | >8% |
| Bootstrap Trim (0-5s) | 10-30% | <10% or >30% | N/A |
| Late Trim (30s+) | <5% | >5% | >10% |
| First Utterance Dupes | <3% | >3% | >5% |
| Post-ASR Latency P95 | <3ms | >3ms | >5ms |
| Context Length P95 | <200 tokens | >200 | >250 |
| Validator Fail Rate | <0.3% | >0.3% | >1.0% |
| PII Error Rate | ~0% | >0.1% | >0.5% |

## 🔍 Troubleshooting

### High False Drop Rate
1. Check post-ASR decision logic
2. Review feature toggle settings
3. Examine validator/PII integration
4. Consider rollback if >1%

### Bootstrap Not Working
1. Check first utterance duplicate rate
2. Review session age bucket efficiency
3. Adjust `boot_min` or `boot_ratio` in code
4. Test with synthetic greeting repeats

### Performance Regression
1. Check P95 latency trends
2. Profile post-ASR decision timing
3. Review context update efficiency
4. Consider optimization or rollback

### Content Quality Issues
1. Run manual content audit
2. Check validator fail rates
3. Review trimming precision/recall
4. Adjust thresholds if systematic bias found

## 📁 File Structure

```
config/
├── hallucination_alerts.yml      # Prometheus alert rules
└── hallucination_dashboard.json  # Grafana dashboard config

scripts/
└── validate_hallucination_metrics.py  # Production validation

utils/
├── hallucination_metrics.py      # Metrics collection framework
└── hallucination_detector.py     # Updated with metrics integration

services/
└── post_asr_pipeline.py         # Updated with metrics integration

test_hallucination_metrics.py    # Comprehensive metrics tests
```

## ✅ Ready for Production

The system is ready for production rollout when:
- [ ] All 10 metric categories are collecting data
- [ ] Validation script passes all checks
- [ ] Alerts are configured and tested
- [ ] Dashboard is deployed and functional
- [ ] Rollback plan is documented and tested
- [ ] Weekly content audit process is established

Run the validation script to confirm readiness:
```bash
python scripts/validate_hallucination_metrics.py
```

If all checks pass with "READY FOR PRODUCTION RAMP" status, proceed with canary deployment.