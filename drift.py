"""Statistical drift detection with multiple-testing control and a multivariate test."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        adj[i] = prev
    return adj


def univariate_drift(ref, cur, alpha=0.05):
    rows = []
    for c in ref.columns:
        r, u = ref[c].dropna().values, cur[c].dropna().values
        ks = ks_2samp(r, u)
        rows.append({
            "Feature": c,
            "KS stat": ks.statistic,
            "p-value": ks.pvalue,
            # effect size: Wasserstein distance in units of reference std
            "Wasserstein (std units)": wasserstein_distance(r, u) / (r.std() + 1e-12),
            "Missing rate": cur[c].isna().mean(),
        })
    df = pd.DataFrame(rows)
    df["p (BH-adjusted)"] = benjamini_hochberg(df["p-value"].values)
    df["Drift flagged"] = (df["p (BH-adjusted)"] < alpha) & (df["Wasserstein (std units)"] > 0.1)
    return df.sort_values("Wasserstein (std units)", ascending=False).reset_index(drop=True)


def classifier_two_sample_test(ref, cur, seed=0, n_estimators=80):
    """Train a classifier to tell reference from current rows. Cross-validated
    AUROC = multivariate drift score (0.5 = indistinguishable).
    Feature importances localise the shift. Note: missingness itself is
    detectable because NaNs are median-imputed to a value the model can spot."""
    n = min(len(ref), len(cur))
    ref = ref.sample(n, random_state=seed)
    cur = cur.sample(n, random_state=seed)
    X = pd.concat([ref, cur]).values
    y = np.r_[np.zeros(n), np.ones(n)]
    X = SimpleImputer(strategy="median").fit_transform(
        np.where(np.isnan(X), np.nan, X))
    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5,
                                 random_state=seed, n_jobs=-1)
    proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=ref.columns).sort_values(ascending=False)
    return auc, imp


def null_drift_score(ref, seed=0):
    """Calibrate the detector: reference vs reference split should give ~0.5."""
    a = ref.sample(frac=0.5, random_state=seed)
    b = ref.drop(a.index)
    return classifier_two_sample_test(a, b, seed)[0]