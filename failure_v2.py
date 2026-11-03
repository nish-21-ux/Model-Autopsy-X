"""Failure investigation v2: paired clean-vs-stressed analysis, error rules, confidence."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text


def paired_failure_breakdown(model, X_clean, X_stress, y):
    y = np.asarray(y)
    ok_c = model.predict(X_clean) == y
    ok_s = model.predict(X_stress) == y
    cats = np.select(
        [ok_c & ok_s, ok_c & ~ok_s, ~ok_c & ok_s, ~ok_c & ~ok_s],
        ["Stable correct", "Newly broken by stress", "Fixed by stress (luck)",
         "Inherent error (wrong even when clean)"], default="unknown")
    return pd.Series(cats).value_counts().rename_axis("Category").reset_index(name="Count"), cats


def feature_error_auc(X_stress, error, fill_values):
    """How well does each feature alone separate erroneous from correct rows?
    0.5 = no relation. Reported as |AUC-0.5|*2 in [0,1] with the direction."""
    Xf = X_stress.fillna(fill_values)
    rows = []
    for c in Xf.columns:
        auc = roc_auc_score(error, Xf[c])
        rows.append({"Feature": c, "AUROC (feature -> error)": auc,
                     "Association strength": abs(auc - 0.5) * 2,
                     "Direction": "higher value -> more errors" if auc > 0.5 else "lower value -> more errors"})
    return pd.DataFrame(rows).sort_values("Association strength", ascending=False).reset_index(drop=True)


def error_rules(X_stress, error, fill_values, max_depth=3, min_leaf=40):
    """Depth-limited tree on the error indicator -> human-readable failure regions."""
    Xf = X_stress.fillna(fill_values)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                  class_weight="balanced", random_state=0)
    tree.fit(Xf, error)
    return export_text(tree, feature_names=list(Xf.columns), show_weights=True)


def confidence_profile(proba, y, high=0.9):
    proba, y = np.asarray(proba), np.asarray(y)
    conf = np.maximum(proba, 1 - proba)
    wrong = (proba >= 0.5).astype(int) != y
    n_wrong = wrong.sum()
    return {
        "errors": int(n_wrong),
        "confident_errors": int((wrong & (conf >= high)).sum()),
        "confident_error_share": float((wrong & (conf >= high)).sum() / max(n_wrong, 1)),
        "mean_conf_correct": float(conf[~wrong].mean()),
        "mean_conf_wrong": float(conf[wrong].mean()) if n_wrong else float("nan"),
    }