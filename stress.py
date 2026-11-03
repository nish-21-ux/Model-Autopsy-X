"""Scale-aware, severity-controlled stress functions.
Severity units are comparable across faults: noise and drift are in units of
each feature's own training standard deviation; missingness is a fraction."""
import numpy as np
import pandas as pd

COMPONENT_ORDER = ["Feature Drift", "Measurement Noise", "Missing Values"]


def apply_drift(X, severity, features, ref_std):
    Xs = X.copy()
    for f in features:
        Xs[f] = Xs[f] + severity * ref_std[f]
    return Xs


def apply_noise(X, severity, seed, ref_std):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, X.shape) * ref_std.values * severity
    return X + noise


def apply_missing(X, severity, seed):
    rng = np.random.default_rng(seed)
    return X.mask(rng.random(X.shape) < severity)


def apply_components(X, active, severities, seed, ref_std, drift_features):
    """Apply any subset of faults in a fixed order. `active` is a set of names."""
    Xs = X.copy()
    if "Feature Drift" in active:
        Xs = apply_drift(Xs, severities["Feature Drift"], drift_features, ref_std)
    if "Measurement Noise" in active:
        Xs = apply_noise(Xs, severities["Measurement Noise"], seed, ref_std)
    if "Missing Values" in active:
        Xs = apply_missing(Xs, severities["Missing Values"], seed + 10_000)
    return Xs