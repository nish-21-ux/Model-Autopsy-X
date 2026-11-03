"""Blind diagnosis benchmark.
Inject a random set of faults, then diagnose them from LABEL-FREE signals only
(unlabeled current data vs reference statistics). Score against the known truth.
Faults: Feature Drift (mean shift), Feature Scaling (multiplier), Measurement Noise,
Missing Values. Injection order: scale -> drift -> noise -> missing."""
import numpy as np
import pandas as pd
from stress import apply_drift, apply_noise, apply_missing

FAULTS = ["Feature Drift", "Feature Scaling", "Measurement Noise", "Missing Values"]


def reference_stats(X_ref, n_cur=None, n_boot=300, seed=0, q=0.995):
    """Reference mean/std plus an EMPIRICAL NULL: how much each signal wobbles when
    n_cur fault-free rows are resampled from the reference. Margins are set from this,
    so skewed or heavy-tailed features automatically get wider margins."""
    mean, std = X_ref.mean(), X_ref.std()
    ref = {"mean": mean, "std": std}
    if n_cur:
        rng = np.random.default_rng(seed)
        A = X_ref.to_numpy()
        ratios, zs, meds = [], [], []
        for _ in range(n_boot):
            S = A[rng.integers(0, len(A), n_cur)]
            ratios.append(S.std(axis=0, ddof=1) / std.values)
            zs.append(np.abs(S.mean(axis=0) - mean.values) / std.values)
            meds.append(np.median((S.std(axis=0, ddof=1) / std.values) ** 2))
        ref["null"] = {"std_ratio_q": pd.Series(np.quantile(ratios, q, axis=0), index=X_ref.columns),
                       "z_q": pd.Series(np.quantile(zs, q, axis=0), index=X_ref.columns),
                       "med_r2_q": float(np.quantile(meds, q))}
    return ref


def random_truth(features, rng, p_fault=0.5):
    t = {f: bool(rng.random() < p_fault) for f in FAULTS}
    t["drift_features"] = list(rng.choice(features, rng.integers(1, 4), replace=False)) if t["Feature Drift"] else []
    t["drift_sev"] = float(rng.uniform(0.3, 2.0)) if t["Feature Drift"] else 0.0
    t["scale_features"] = list(rng.choice(features, rng.integers(1, 3), replace=False)) if t["Feature Scaling"] else []
    t["scale_factor"] = float(rng.uniform(1.4, 2.2)) if t["Feature Scaling"] else 1.0
    t["noise_sev"] = float(rng.uniform(0.2, 1.5)) if t["Measurement Noise"] else 0.0
    t["missing_sev"] = float(rng.uniform(0.05, 0.4)) if t["Missing Values"] else 0.0
    return t


def inject(X, truth, ref_std, seed):
    Xs = X.copy()
    for f in truth["scale_features"]:
        Xs[f] = Xs[f] * truth["scale_factor"]
    if truth["Feature Drift"]:
        Xs = apply_drift(Xs, truth["drift_sev"], truth["drift_features"], ref_std)
    if truth["Measurement Noise"]:
        Xs = apply_noise(Xs, truth["noise_sev"], seed, ref_std)
    if truth["Missing Values"]:
        Xs = apply_missing(Xs, truth["missing_sev"], seed + 10_000)
    return Xs


def compute_signals(ref, X_cur):
    """Label-free signals. ref = reference_stats(X_ref)."""
    return {
        "n": int(len(X_cur)),
        "features": list(X_cur.columns),
        "missing_rate": float(X_cur.isna().to_numpy().mean()),
        "std_ratio": (X_cur.std() / ref["std"]).to_dict(),
        "mean_shift_z": ((X_cur.mean() - ref["mean"]) / ref["std"]).to_dict(),
    }


def diagnose(sig, ref):
    """Rule-based estimator built on the signals above (hand-derived, no labels)."""
    feats, n, d = sig["features"], sig["n"], len(sig["features"])
    r2 = np.array([sig["std_ratio"][f] ** 2 for f in feats])
    nl = ref.get("null")
    thr_var = 1.1 * max(nl["med_r2_q"] - 1.0, 0.03) if nl else 6 * np.sqrt(2 / n) / np.sqrt(d)
    thr_scale_j = {f: (max(1.15, 1.1 * nl["std_ratio_q"][f]) if nl else 1 + max(0.15, 4 / np.sqrt(2 * n)))
                   for f in feats}
    thr_drift_j = {f: (max(0.15, 1.1 * nl["z_q"][f]) if nl else max(0.2, 4 / np.sqrt(n))) for f in feats}

    out = {f: {"present": False, "severity": 0.0, "features": []} for f in FAULTS}
    out["Missing Values"].update(present=sig["missing_rate"] > 0.02, severity=sig["missing_rate"])

    s2 = max(np.median(r2) - 1.0, 0.0)                    # median is robust to a few scaled features
    noise_present = s2 > thr_var
    s2_used = s2 if noise_present else 0.0
    out["Measurement Noise"].update(present=bool(noise_present), severity=float(np.sqrt(s2_used)))

    f_hat = {f: float(np.sqrt(max(sig["std_ratio"][f] ** 2 - s2_used, 0.0))) for f in feats}
    scaled = [f for f in feats if f_hat[f] > thr_scale_j[f]]
    out["Feature Scaling"].update(present=bool(scaled), features=scaled,
                                  severity=float(np.mean([f_hat[f] for f in scaled])) if scaled else 0.0)

    resid = {}
    for f in feats:
        k = f_hat[f] if f in scaled else 1.0
        # after scaling by k, expected mean shift is (k-1)*mu/sigma; remainder is drift
        resid[f] = sig["mean_shift_z"][f] - (k - 1.0) * ref["mean"][f] / ref["std"][f]
    def _thr(f):
        # error in the estimated scale factor k leaks into the drift residual in
        # proportion to |mean/std|: widen the margin for scaled features accordingly
        if f not in scaled:
            return thr_drift_j[f]
        return thr_drift_j[f] + 3 * 1.5 * f_hat[f] / np.sqrt(2 * n) * abs(ref["mean"][f] / ref["std"][f])
    drifted = [f for f in feats if abs(resid[f]) > _thr(f)]
    out["Feature Drift"].update(present=bool(drifted), features=drifted,
                                severity=float(np.mean([abs(resid[f]) for f in drifted])) if drifted else 0.0)
    return out


def _true_sev(t, fault):
    return {"Feature Drift": t["drift_sev"], "Feature Scaling": t["scale_factor"],
            "Measurement Noise": t["noise_sev"], "Missing Values": t["missing_sev"]}[fault]


def _true_feats(t, fault):
    return {"Feature Drift": t["drift_features"], "Feature Scaling": t["scale_features"]}.get(fault, [])


def run_benchmark(X_ref, X_pool, n_trials=300, seed=0, p_fault=0.5):
    rng = np.random.default_rng(seed)
    ref = reference_stats(X_ref, n_cur=len(X_pool), seed=seed)
    feats = list(X_pool.columns)
    rows = []
    for k in range(n_trials):
        t = random_truth(feats, rng, p_fault)
        Xs = inject(X_pool, t, ref["std"], seed=int(rng.integers(1_000_000)))
        diag = diagnose(compute_signals(ref, Xs), ref)
        row = {"trial": k, "n_true_faults": sum(t[f] for f in FAULTS)}
        for f in FAULTS:
            tf, pf = t[f], diag[f]["present"]
            row[f"{f}|true"], row[f"{f}|pred"] = tf, pf
            row[f"{f}|true_sev"], row[f"{f}|est_sev"] = _true_sev(t, f), diag[f]["severity"]
            tfeat, pfeat = set(_true_feats(t, f)), set(diag[f]["features"])
            row[f"{f}|jaccard"] = len(tfeat & pfeat) / len(tfeat | pfeat) if (tf and pf and tfeat | pfeat) else np.nan
        row["exact_set"] = all(row[f"{f}|true"] == row[f"{f}|pred"] for f in FAULTS)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(trials):
    out = []
    for f in FAULTS:
        t, p = trials[f"{f}|true"], trials[f"{f}|pred"]
        tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        hit = trials[t & p]
        out.append({"Fault": f, "Trials with fault": int(t.sum()), "TP": tp, "FP": fp, "FN": fn,
                    "Precision": prec, "Recall": rec,
                    "F1": 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else np.nan,
                    "Severity MAE": float((hit[f"{f}|est_sev"] - hit[f"{f}|true_sev"]).abs().mean()) if len(hit) else np.nan,
                    "Feature Jaccard": float(hit[f"{f}|jaccard"].mean()) if f in ("Feature Drift", "Feature Scaling") and len(hit) else np.nan})
    clean = trials[trials.n_true_faults == 0]
    clean_fp = float((~clean[[f"{f}|pred" for f in FAULTS]].eq(False).all(axis=1)).mean()) if len(clean) else np.nan
    overall = {"Exact fault-set accuracy": float(trials.exact_set.mean()),
               "False-alarm rate on fault-free trials": clean_fp, "Fault-free trials": int(len(clean)),
               "Trials": int(len(trials))}
    return pd.DataFrame(out), overall


def recall_by_severity(trials, fault, bins=3):
    d = trials[trials[f"{fault}|true"]].copy()
    d["bin"] = pd.qcut(d[f"{fault}|true_sev"], bins, duplicates="drop")
    return d.groupby("bin", observed=True)[f"{fault}|pred"].mean().rename("Recall").reset_index().assign(Fault=fault)