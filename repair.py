"""Repair simulator. Honest protocol: stressed data is split into a REPAIR set (what you
could realistically label or observe after the incident) and a HELD-OUT set used only
for scoring. Each repair is compared to doing nothing with a paired bootstrap."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from metrics import paired_bootstrap, expected_calibration_error
from sklearn.metrics import brier_score_loss


def _fit(model, X, y, w=None):
    m = clone(model)
    if w is None:
        return m.fit(X, y)
    return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": np.asarray(w)})


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def simulate_repairs(model, X_train, y_train, X_clean, X_stress, y, drop_features=None,
                     repair_frac=0.3, seed=0):
    idx = np.arange(len(y))
    rep_i, held_i = train_test_split(idx, train_size=repair_frac, stratify=y, random_state=seed)
    y = np.asarray(y)
    Xr, yr = X_stress.iloc[rep_i], y[rep_i]
    Xh, yh = X_stress.iloc[held_i], y[held_i]
    auc_clean_h = roc_auc_score(yh, model.predict_proba(X_clean.iloc[held_i])[:, 1])
    base_p = model.predict_proba(Xh)[:, 1]
    auc_base = roc_auc_score(yh, base_p)
    probs, notes = {"Do nothing": base_p}, {"Do nothing": "baseline"}

    # 1. recalibrate (Platt on the repair set): fixes confidence, not ranking
    cal = LogisticRegression().fit(_logit(model.predict_proba(Xr)[:, 1]).reshape(-1, 1), yr)
    probs["Recalibrate (Platt)"] = cal.predict_proba(_logit(base_p).reshape(-1, 1))[:, 1]
    notes["Recalibrate (Platt)"] = f"uses {len(yr)} labelled stressed rows"

    # 2. retrain with labelled stressed rows up-weighted
    w = np.r_[np.ones(len(X_train)), np.full(len(Xr), 5.0)]
    m2 = _fit(model, pd.concat([X_train, Xr]), np.r_[np.asarray(y_train), yr], w)
    probs["Retrain on train + stressed sample"] = m2.predict_proba(Xh)[:, 1]
    notes["Retrain on train + stressed sample"] = f"uses {len(yr)} labelled stressed rows (weight 5x)"

    # 3. importance reweighting - LABEL-FREE (needs only unlabeled stressed X)
    dom_X = pd.concat([X_train, X_stress.iloc[rep_i]])
    dom_y = np.r_[np.zeros(len(X_train)), np.ones(len(rep_i))]
    dom = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed)
    pd_ = cross_val_predict(dom, dom_X, dom_y, cv=3, method="predict_proba")[:len(X_train), 1]
    wt = np.clip(pd_ / (1 - pd_ + 1e-6) * (len(X_train) / len(rep_i)), 0.05, 20)
    ess = wt.sum() ** 2 / (wt ** 2).sum()
    m3 = _fit(model, X_train, y_train, wt)
    probs["Importance reweighting (label-free)"] = m3.predict_proba(Xh)[:, 1]
    notes["Importance reweighting (label-free)"] = f"no labels needed; effective sample size {ess:.0f}/{len(X_train)}"

    # 4. drop suspected features and retrain
    if drop_features:
        keep = [c for c in X_train.columns if c not in drop_features]
        m4 = _fit(model, X_train[keep], y_train)
        probs["Drop suspect features"] = m4.predict_proba(Xh[keep])[:, 1]
        notes["Drop suspect features"] = f"removed {', '.join(drop_features)}"

    rows = []
    for name, p in probs.items():
        auc = roc_auc_score(yh, p)
        if name == "Do nothing":
            d, lo, hi = 0.0, 0.0, 0.0
        else:
            d, lo, hi, _ = paired_bootstrap(yh, p, base_p, "AUROC", n=500, seed=seed)
        lost = auc_clean_h - auc_base
        rows.append({"Repair": name, "AUROC": auc, "Gain vs nothing": d, "95% CI of gain": f"[{lo:+.3f}, {hi:+.3f}]",
                     "Recovered share": (auc - auc_base) / lost if lost > 1e-6 else np.nan,
                     "Brier": brier_score_loss(yh, np.clip(p, 1e-6, 1 - 1e-6)),
                     "ECE": expected_calibration_error(yh, p),
                     "Significant gain": bool(lo > 0), "Notes": notes[name]})
    table = pd.DataFrame(rows)
    return table, {"auc_clean_heldout": auc_clean_h, "auc_stressed_heldout": auc_base,
                   "n_repair": len(rep_i), "n_heldout": len(held_i)}


def abstention_table(proba, y, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Selective prediction: answer only the most confident rows."""
    proba, y = np.asarray(proba), np.asarray(y)
    conf = np.maximum(proba, 1 - proba)
    order = np.argsort(-conf)
    rows = []
    for c in coverages:
        keep = order[: max(int(round(c * len(y))), 1)]
        rows.append({"Coverage": c, "Accuracy on answered": float(((proba[keep] >= .5).astype(int) == y[keep]).mean()),
                     "Rows answered": len(keep)})
    return pd.DataFrame(rows)