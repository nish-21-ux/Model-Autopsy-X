"""Reliability metrics with uncertainty."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss, log_loss,
                             balanced_accuracy_score, matthews_corrcoef)


def expected_calibration_error(y, p, n_bins=10):
    """Adaptive (equal-mass) binning ECE for binary P(y=1)."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        ece += len(chunk) / len(y) * abs(y[chunk].mean() - p[chunk].mean())
    return ece


def reliability_curve(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, n_bins):
        rows.append((p[chunk].mean(), y[chunk].mean()))
    return np.array(rows)


def metric_suite(y, proba):
    y = np.asarray(y)
    p = np.clip(np.asarray(proba), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y, p),
        "Accuracy": accuracy_score(y, pred),
        "Balanced Acc": balanced_accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, p),
        "ECE": expected_calibration_error(y, p),
    }


METRIC_FUNCS = {
    "AUROC": lambda y, p: roc_auc_score(y, p),
    "Accuracy": lambda y, p: accuracy_score(y, (p >= 0.5).astype(int)),
    "F1": lambda y, p: f1_score(y, (p >= 0.5).astype(int), zero_division=0),
    "Brier": lambda y, p: brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
}


def bootstrap_ci(y, p, metric="AUROC", n=500, seed=0, alpha=0.05):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    fn = METRIC_FUNCS[metric]
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return fn(y, p), lo, hi


def paired_bootstrap(y, p_a, p_b, metric="AUROC", n=1000, seed=0):
    """Same resampled rows for both models. Returns mean diff (A-B), 95% CI, P(A>B)."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    fn = METRIC_FUNCS[metric]
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(fn(y[i], p_a[i]) - fn(y[i], p_b[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, (d > 0).mean()


def degradation_auc(severities, values):
    """Area under the metric-vs-severity curve, normalised by severity range
    (so it is the *average* metric over the sweep; higher = more robust)."""
    s, v = np.asarray(severities, float), np.asarray(values, float)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return trap(v, s) / (s.max() - s.min())


def breaking_point(severities, values, threshold):
    """First severity (linear interpolation) where the metric drops below threshold."""
    for k in range(1, len(severities)):
        if values[k] < threshold <= values[k - 1]:
            s0, s1, v0, v1 = severities[k-1], severities[k], values[k-1], values[k]
            return s0 + (v0 - threshold) / (v0 - v1) * (s1 - s0)
    return None