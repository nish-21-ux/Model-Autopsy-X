"""Evidence packets, a template Analyst, a faithfulness checker, and training-data generation."""
import json, re
import numpy as np
import pandas as pd
from blind_benchmark import (FAULTS, reference_stats, random_truth, inject, compute_signals, diagnose)


def build_packet(sig, diag):
    return {
        "n_rows": sig["n"], "n_features": len(sig["features"]),
        "n_faults_detected": sum(bool(d["present"]) for d in diag.values()),
        "missing_rate": round(sig["missing_rate"], 4),
        "std_ratio": {f: round(v, 3) for f, v in sig["std_ratio"].items()},
        "mean_shift_z": {f: round(v, 3) for f, v in sig["mean_shift_z"].items()},
        "diagnosis": {f: {"present": d["present"], "severity": round(d["severity"], 3),
                          "features": d["features"], "n_features": len(d["features"])} for f, d in diag.items()},
    }


_UNIT = {"Feature Drift": "std units", "Feature Scaling": "x multiplier",
         "Measurement Noise": "x feature std", "Missing Values": "fraction missing"}


def render_analysis(packet):
    """Deterministic template Analyst. Every number comes straight from the packet."""
    d = packet["diagnosis"]
    found = [f for f in FAULTS if d[f]["present"]]
    if not found:
        return ("No fault detected. Label-free signals (missing rate "
                f"{packet['missing_rate']}) are within the range expected from sampling variation "
                f"for {packet['n_rows']} rows. This does not rule out a concept change that leaves "
                "inputs unchanged; check labelled performance when labels arrive.")
    lines = [f"Detected {len(found)} fault(s) from label-free evidence on {packet['n_rows']} rows:"]
    for f in found:
        s = d[f]["severity"]
        extra = ""
        if d[f]["features"]:
            top = sorted(d[f]["features"], key=lambda c: -abs(packet["mean_shift_z"][c]) if f == "Feature Drift"
                         else -packet["std_ratio"][c])[:3]
            extra = f" Affected features: {', '.join(top)}."
        lines.append(f"- {f}: estimated severity {s} ({_UNIT[f]}).{extra}")
    lines.append("Limits: severities are estimates; drift and scaling are hard to separate on features "
                 "whose mean is large relative to their spread, and a change in P(y|x) cannot be seen "
                 "without labels.")
    return "\n".join(lines)


def _leaves(o):
    if isinstance(o, bool):
        return []
    if isinstance(o, (int, float, np.integer, np.floating)):
        return [float(o)]
    if isinstance(o, dict):
        return [x for v in o.values() for x in _leaves(v)]
    if isinstance(o, (list, tuple)):
        return [x for v in o for x in _leaves(v)]
    return []


def verify_numbers(text, packet):
    """Faithfulness check: every number in `text` must match some packet value at the
    precision it is written with (or its percentage form). Returns (rate, unsupported)."""
    vals = _leaves(packet)
    clean = re.sub(r"Feature_\d+", "", text)
    toks = re.findall(r"-?\d+(?:\.\d+)?%?", clean)
    bad, ok = [], 0
    for t in toks:
        pct = t.endswith("%")
        x = float(t.rstrip("%")); dec = len(t.rstrip("%").split(".")[1]) if "." in t else 0
        good = any(round(v * (100 if pct else 1), dec) == round(x, dec) for v in vals)
        ok += good
        if not good:
            bad.append(t)
    return (ok / len(toks) if toks else 1.0), bad


def training_example(sig, diag, truth):
    """Instruction = evidence packet. Response is written from the GROUND TRUTH (not from the rule),
    so the model is trained to name real causes; undetected true faults are reported as weak evidence."""
    packet = build_packet(sig, diag)
    parts = []
    for f in FAULTS:
        if truth[f] and diag[f]["present"]:
            parts.append(f"{f}: confirmed, estimated severity {packet['diagnosis'][f]['severity']} ({_UNIT[f]}).")
        elif truth[f]:
            parts.append(f"{f}: possibly present but evidence is below the detection margin; cannot be confirmed.")
    if not parts:
        resp = ("No fault detected. Signals are consistent with sampling variation; "
                "this does not rule out a change in P(y|x) that inputs cannot reveal.")
    else:
        resp = "Findings from label-free evidence:\n" + "\n".join(parts) + \
               "\nRecommended next step: confirm with labelled data, then test a repair on held-out stressed data."
    return {"instruction": "Interpret this evidence packet for an ML model reliability investigation. "
                           "Name only faults the evidence supports.\n" + json.dumps(packet),
            "response": resp}


def generate_training_set(X_ref, X_pool, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ref = reference_stats(X_ref, n_cur=len(X_pool), seed=seed)
    rows = []
    for _ in range(n):
        t = random_truth(list(X_pool.columns), rng)
        Xs = inject(X_pool, t, ref["std"], seed=int(rng.integers(1_000_000)))
        sig = compute_signals(ref, Xs)
        rows.append(training_example(sig, diagnose(sig, ref), t))
    return pd.DataFrame(rows)