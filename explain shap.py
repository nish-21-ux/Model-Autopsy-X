"""Optional SHAP layer. Requires `pip install shap`. NOTE: written against the
documented SHAP API but not executed in my sandbox (shap not installable there) -
run it once and check the plot before relying on it."""
import numpy as np
import pandas as pd


def mean_abs_shap(pipeline, X, max_rows=300, seed=0):
    import shap
    Xs = X.sample(min(max_rows, len(X)), random_state=seed)
    steps = pipeline.steps
    imputer = steps[0][1]
    clf = steps[-1][1]
    Xt = imputer.transform(Xs)
    if len(steps) == 3:                      # logistic regression path (scaler)
        Xt = steps[1][1].transform(Xs.pipe(lambda d: pd.DataFrame(imputer.transform(d), columns=d.columns)))
        expl = shap.LinearExplainer(clf, Xt)
        vals = expl.shap_values(Xt)
    else:                                    # tree models
        expl = shap.TreeExplainer(clf)
        vals = expl.shap_values(Xt)
    if isinstance(vals, list):               # older SHAP: list per class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:                       # newer SHAP: (rows, features, classes)
        vals = vals[:, :, 1]
    return pd.Series(np.abs(vals).mean(axis=0), index=X.columns)