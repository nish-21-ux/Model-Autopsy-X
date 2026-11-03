"""Interventional (Shapley) attribution of a performance drop to injected faults."""
from itertools import combinations
from math import factorial
import numpy as np
from sklearn.metrics import roc_auc_score
from stress import apply_components


def shapley_fault_attribution(model, X, y, components, severities, ref_std,
                              drift_features, n_seeds=5):
    """Apply every subset of `components` to clean X, measure AUROC, and
    apportion the total AUROC drop with exact Shapley values.
    Returns (shares dict, total_drop, subset_table dict)."""
    cache = {}

    def perf(subset):
        key = frozenset(subset)
        if key not in cache:
            vals = []
            for s in range(n_seeds):
                Xs = apply_components(X, key, severities, s, ref_std, drift_features)
                vals.append(roc_auc_score(y, model.predict_proba(Xs)[:, 1]))
            cache[key] = float(np.mean(vals))
        return cache[key]

    n = len(components)
    base = perf(set())
    loss = lambda S: base - perf(S)
    phi = {}
    for c in components:
        others = [o for o in components if o != c]
        total = 0.0
        for k in range(len(others) + 1):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                total += w * (loss(set(S) | {c}) - loss(set(S)))
        phi[c] = total
    total_drop = loss(set(components))
    table = {tuple(sorted(k)): v for k, v in cache.items()}
    return phi, total_drop, table


def feature_restoration(model, X_clean, X_stress, y, features):
    """Causal check of a diagnosis: put the named features back to their clean
    values inside the stressed data and measure how much AUROC is recovered."""
    auc_clean = roc_auc_score(y, model.predict_proba(X_clean)[:, 1])
    auc_stress = roc_auc_score(y, model.predict_proba(X_stress)[:, 1])
    Xr = X_stress.copy()
    Xr[features] = X_clean[features]
    auc_restored = roc_auc_score(y, model.predict_proba(Xr)[:, 1])
    lost = auc_clean - auc_stress
    recovered = (auc_restored - auc_stress) / lost if lost > 1e-9 else np.nan
    return auc_clean, auc_stress, auc_restored, recovered