
import streamlit as st
import requests
import os
import json


# Your Colab-hosted trained Qwen model
API_URL = "https://contented-cringing-postbox.ngrok-free.dev/diagnose"


def build_autopsy_findings(evidence):
    findings = []

    baseline = evidence.get("baseline_score")
    stressed = evidence.get("stressed_score")

    if baseline is not None and stressed is not None:
        degradation = baseline - stressed

        if degradation >= 0.20:
            severity = "Critical"
            finding = "Major performance degradation detected."
        elif degradation >= 0.05:
            severity = "Warning"
            finding = "Moderate performance degradation detected."
        else:
            severity = "Stable"
            finding = "No substantial performance degradation detected."

        findings.append({
            "category": "Performance",
            "severity": severity,
            "finding": finding,
            "evidence": f"Score decreased from {baseline:.3f} to {stressed:.3f}.",
            "recommendation": "Investigate performance under the affected conditions."
        })

    drift_features = evidence.get("drift_features", [])

    if drift_features:
        findings.append({
            "category": "Data Drift",
            "severity": "Warning",
            "finding": "Feature distribution changes were detected.",
            "evidence": "Affected features: " + ", ".join(map(str, drift_features)),
            "recommendation": "Investigate changes in incoming feature distributions."
        })

    missing_rate = evidence.get("missing_rate")

    if missing_rate is not None:
        severity = (
            "Critical" if missing_rate >= 0.20
            else "Warning" if missing_rate >= 0.05
            else "Stable"
        )

        findings.append({
            "category": "Data Quality",
            "severity": severity,
            "finding": "Missing-value analysis completed.",
            "evidence": f"Missing-value rate: {missing_rate:.1%}.",
            "recommendation": "Investigate missing-data patterns and preprocessing."
        })

    ece = evidence.get("ece")

    if ece is not None:
        severity = (
            "Critical" if ece >= 0.20
            else "Warning" if ece >= 0.10
            else "Stable"
        )

        findings.append({
            "category": "Calibration",
            "severity": severity,
            "finding": "Prediction calibration was evaluated.",
            "evidence": f"Expected Calibration Error: {ece:.4f}.",
            "recommendation": "Investigate probability calibration."
        })

    confident_errors = evidence.get("confident_errors")

    if confident_errors is not None:
        findings.append({
            "category": "Prediction Reliability",
            "severity": "Warning" if confident_errors > 0 else "Stable",
            "finding": "High-confidence prediction errors were evaluated.",
            "evidence": f"Number of confident errors: {confident_errors}.",
            "recommendation": "Investigate incorrect high-confidence predictions."
        })

    if not findings:
        findings.append({
            "category": "Investigation",
            "severity": "Unavailable",
            "finding": "Insufficient evidence for an investigation.",
            "evidence": "No supported audit measurements were provided.",
            "recommendation": "Run model evaluation and stress testing first."
        })

    return findings


def generate_investigation_summary(findings):
    critical = sum(f["severity"] == "Critical" for f in findings)
    warnings = sum(f["severity"] == "Warning" for f in findings)
    stable = sum(f["severity"] == "Stable" for f in findings)

    if critical:
        verdict = "HIGH RISK"
        explanation = "Critical reliability concerns were identified."
    elif warnings:
        verdict = "NEEDS ATTENTION"
        explanation = "Some reliability concerns require investigation."
    elif stable:
        verdict = "NO MAJOR ISSUES DETECTED"
        explanation = "No major issues were detected in the available evidence."
    else:
        verdict = "INSUFFICIENT EVIDENCE"
        explanation = "The available evidence is insufficient for a verdict."

    return {
        "verdict": verdict,
        "explanation": explanation,
        "critical": critical,
        "warnings": warnings,
        "stable": stable
    }


def generate_qwen_diagnosis(evidence, findings):
    api_key = os.getenv("MODELAUTOPSY_API_KEY")

    if not api_key:
        return None, "API key not configured."

    prompt = f"""
You are the AI investigation assistant inside Model Autopsy X.

Analyze the supplied machine learning reliability evidence.

RULES:
1. Use only the supplied measurements and findings.
2. Never invent metrics, observations, or test results.
3. Distinguish observed evidence from possible causes.
4. Do not claim a cause is proven without supporting evidence.
5. Give practical investigation steps and mitigation recommendations.
6. If evidence is insufficient, explicitly say so.

AUDIT EVIDENCE:
{json.dumps(evidence, indent=2, default=str)}

RULE-BASED FINDINGS:
{json.dumps(findings, indent=2, default=str)}

Provide:
- Overall risk assessment
- Key evidence
- Possible contributing factors
- Recommended investigation steps
- Mitigation recommendations
- Limitations and uncertainties
"""

    try:
        response = requests.post(
            API_URL,
            json={"prompt": prompt},
            headers={"x-api-key": api_key},
            timeout=180
        )

        response.raise_for_status()

        result = response.json()
        diagnosis = result.get("diagnosis")

        if not diagnosis:
            return None, "The model returned an empty diagnosis."

        return diagnosis, None

    except requests.exceptions.Timeout:
        return None, "The model API timed out. Try again."

    except requests.exceptions.RequestException as error:
        return None, f"Model API connection failed: {error}"


def render_autopsy_intelligence(evidence):
    st.markdown("---")
    st.header("12. Autopsy Intelligence Engine")

    st.caption(
        "Evidence-based investigation with rule-based analysis "
        "and a fine-tuned Qwen AI diagnosis."
    )

    findings = build_autopsy_findings(evidence)
    summary = generate_investigation_summary(findings)

    st.subheader("Investigation Verdict")
    st.markdown(f"### {summary['verdict']}")
    st.write(summary["explanation"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Critical Findings", summary["critical"])
    col2.metric("Warnings", summary["warnings"])
    col3.metric("Stable Findings", summary["stable"])

    st.markdown("---")
    st.subheader("Rule-Based Investigation Findings")

    for index, finding in enumerate(findings, start=1):
        severity = finding["severity"]

        with st.expander(
            f"{index}. {finding['category']} — {severity}",
            expanded=(severity == "Critical")
        ):
            st.markdown(f"#### {finding['finding']}")
            st.write("**Evidence:**", finding["evidence"])
            st.write("**Recommended action:**", finding["recommendation"])

    st.markdown("---")
    st.subheader("Qwen AI Investigation")

    st.caption(
        "AI-generated advisory analysis. Numerical audit findings "
        "remain based on the original evaluation engine."
    )

    if st.button("Generate AI-Assisted Diagnosis"):
        with st.spinner("Qwen is analyzing the audit evidence..."):
            diagnosis, error = generate_qwen_diagnosis(evidence, findings)

        if error:
            st.error(error)
        else:
            st.markdown("### AI Diagnosis")
            st.markdown(diagnosis)

    st.markdown("---")
    st.caption(
        "Important: Rule-based findings and AI-generated explanations "
        "are diagnostic aids, not proof of causal relationships "
        "or deployment safety."
    )

    return findings

