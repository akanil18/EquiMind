# EquiMind v1.0: Self-Reflection & Multi-Domain Adaptability

EquiMind v1.0 includes post-hoc self-critique capabilities and multi-domain reusability.

---

## 🪞 Self-Reflection & Calibration Engine (`equimind.reflection`)

The `SelfReflectionAgent` performs post-hoc evaluations on historical recommendations against actual price outcomes after $N$ days:

- **Bias Detection**: Flags over-optimism/excess bullish bias if a `BUY` rating was followed by a price decline, or over-pessimism if a `SELL` rating was followed by a rally.
- **Conviction Score Calibration**: Calculates a system-wide calibration factor ($>1.0$ boost for high accuracy, $<1.0$ dampener for low accuracy) that fine-tunes future `JudgeAgent` conviction scoring.

---

## 🌐 Multi-Domain Adaptability Suite (`equimind.domain_adapter`)

The underlying orchestration architecture (Reasoning Planner -> Evidence Graph -> Context Optimization Compressor -> Adversarial Committee Debate -> Hierarchical Memory) is generalized for non-financial research domains:

1. **`LegalResearchAdapter`**: Case law precedent analysis, statutory antitrust evaluation, plaintiff vs defense adversarial debate, Supreme Court & Federal Register citations.
2. **`MedicalReviewAdapter`**: Clinical trial Phase III RCT literature synthesis, FDA approval reports, efficacy vs adverse event debate.
3. **`CybersecurityThreatAdapter`**: Vulnerability threat intel, NIST NVD CVSS v3.1 scoring, CISA KEV listing checks, active exploit probability debate.
