
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import hashlib

from dataset_pipeline import prepare_uploaded_dataset

from autopsy_intelligence import render_autopsy_intelligence

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from data_loader import DATASETS, load_split
from dataset_pipeline import prepare_uploaded_dataset
from theme import apply_theme
from stress import apply_components, COMPONENT_ORDER
from metrics import (
    metric_suite, bootstrap_ci, paired_bootstrap, degradation_auc,
    breaking_point, reliability_curve
)
from attribution import shapley_fault_attribution, feature_restoration
from drift import (
    univariate_drift, classifier_two_sample_test, null_drift_score
)
from failure_v2 import (
    paired_failure_breakdown, feature_error_auc, error_rules,
    confidence_profile
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Model Autopsy X",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("MODEL AUTOPSY X")
apply_theme()

st.title("🧬 MODEL AUTOPSY X")
st.subheader("The AI That Investigates Other AI")

st.caption(
    "Every number below carries uncertainty. Every diagnosis is checked "
    "by intervention, because the faults are injected and therefore known."
)

# ==========================================
# DATA & MODELS
# ==========================================

st.sidebar.markdown("---")
st.sidebar.header("Dataset Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload your ML dataset (CSV)",
    type=["csv"],
    key="model_autopsy_dataset"
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        if uploaded_df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        target_column = st.sidebar.selectbox(
            "Select target column",
            options=uploaded_df.columns.tolist()
        )

        try:
            X_train, X_test, y_train, y_test, dataset_metadata = (
                prepare_uploaded_dataset(uploaded_df, target_column)
            )
            DS_NAME = f"Uploaded: {uploaded_file.name}"

        except ValueError as e:
            st.error(f"Dataset preparation failed: {e}")
            st.stop()

    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

else:
    DS_NAME = st.sidebar.selectbox(
        "Choose built-in dataset",
        DATASETS
    )

    @st.cache_data
    def generate_data(name):
        return load_split(name)

    X_train, X_test, y_train, y_test = generate_data(DS_NAME)
    dataset_metadata = {
        "source": "Built-in dataset",
        "target": "Existing dataset target"
    }

REF_STD = X_train.std().fillna(0)
REF_MEDIAN = X_train.median()


@st.cache_resource
def train_models(dataset_key, X, y):
    models = {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=150,
                random_state=42
            )
        ),

        "Extra Trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=150,
                random_state=42
            )
        ),

        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),
    }

    for model in models.values():
        model.fit(X, y)

    return models


models = train_models(
    DS_NAME,
    X_train,
    y_train
)

rf_importance = pd.Series(
    models["Random Forest"][-1].feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

st.sidebar.success(f"Dataset ready: {DS_NAME}")
st.sidebar.caption(
    f"Training rows: {len(X_train)} | "
    f"Testing rows: {len(X_test)} | "
    f"Features: {X_train.shape[1]}"
)


# ==========================================
# SIDEBAR EXPERIMENT SETTINGS
# ==========================================

st.sidebar.header("Experiment settings")

st.sidebar.caption(
    "Faults can be combined. Severities are in comparable units."
)

active = set()
severities = {}

if st.sidebar.checkbox("Feature Drift", value=True):
    active.add("Feature Drift")

severities["Feature Drift"] = st.sidebar.slider(
    "Drift shift (× feature std)",
    0.0, 3.0, 1.5, 0.1
)

drift_features = st.sidebar.multiselect(
    "Drifted features (default: 2 most important)",
    list(X_train.columns),
    default=list(rf_importance.index[:2])
)

if st.sidebar.checkbox("Measurement Noise", value=True):
    active.add("Measurement Noise")

severities["Measurement Noise"] = st.sidebar.slider(
    "Noise σ (× feature std)",
    0.0, 2.0, 0.5, 0.1
)

if st.sidebar.checkbox("Missing Values", value=False):
    active.add("Missing Values")

severities["Missing Values"] = st.sidebar.slider(
    "Missing fraction",
    0.0, 0.5, 0.15, 0.05
)

SEED = st.sidebar.number_input(
    "Random seed",
    0, 10_000, 0
)

# ==========================================
# MODEL SELECTION
# ==========================================

selected_model_name = st.sidebar.selectbox(
    "Model under investigation",
    list(models.keys())
)


# ==========================================
# APPLY STRESS CONDITIONS
# ==========================================

try:
    X_stress = apply_components(
        X_test.copy(),
        active,
        severities,
        SEED,
        REF_STD,
        drift_features
    )

    scenario_label = (
        " + ".join(
            c for c in COMPONENT_ORDER if c in active
        )
        or "Clean Data"
    )

    proba_clean = {
        name: model.predict_proba(X_test)[:, 1]
        for name, model in models.items()
    }

    proba_stress = {
        name: model.predict_proba(X_stress)[:, 1]
        for name, model in models.items()
    }

except Exception as e:
    st.error(f"Stress testing failed: {e}")
    st.stop()


st.info(
    f"**Scenario:** {scenario_label} | "
    f"Seed: {SEED} | "
    f"Test rows: {len(y_test)} | "
    f"Drifted features: "
    f"{', '.join(drift_features) if 'Feature Drift' in active else '-'}"
)

# ==========================================
# 01 PERFORMANCE WITH UNCERTAINTY
# ==========================================

st.divider()
st.header("01 | Performance with Uncertainty")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    brier_score_loss,
    log_loss
)

performance_rows = []

for model_name, model in models.items():

    clean_prob = proba_clean[model_name]
    stress_prob = proba_stress[model_name]

    clean_pred = (clean_prob >= 0.5).astype(int)
    stress_pred = (stress_prob >= 0.5).astype(int)

    performance_rows.append({
        "Model": model_name,
        "Condition": "Clean",
        "Accuracy": accuracy_score(y_test, clean_pred),
        "Precision": precision_score(
            y_test, clean_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, clean_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, clean_pred, zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, clean_pred
        ),
        "AUROC": roc_auc_score(y_test, clean_prob),
        "MCC": matthews_corrcoef(y_test, clean_pred),
        "Brier Score": brier_score_loss(y_test, clean_prob),
        "Log Loss": log_loss(
            y_test, clean_prob, labels=[0, 1]
        )
    })

    performance_rows.append({
        "Model": model_name,
        "Condition": "Stressed",
        "Accuracy": accuracy_score(y_test, stress_pred),
        "Precision": precision_score(
            y_test, stress_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, stress_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, stress_pred, zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, stress_pred
        ),
        "AUROC": roc_auc_score(y_test, stress_prob),
        "MCC": matthews_corrcoef(y_test, stress_pred),
        "Brier Score": brier_score_loss(y_test, stress_prob),
        "Log Loss": log_loss(
            y_test, stress_prob, labels=[0, 1]
        )
    })

performance_df = pd.DataFrame(performance_rows)

st.subheader("Model Performance Comparison")

st.dataframe(
    performance_df.round(4),
    use_container_width=True,
    hide_index=True
)

selected_performance = performance_df[
    performance_df["Model"] == selected_model_name
]

col1, col2, col3, col4 = st.columns(4)

clean_metrics = selected_performance[
    selected_performance["Condition"] == "Clean"
].iloc[0]

stress_metrics = selected_performance[
    selected_performance["Condition"] == "Stressed"
].iloc[0]

col1.metric(
    "Clean Accuracy",
    f"{clean_metrics['Accuracy']:.3f}"
)

col2.metric(
    "Stressed Accuracy",
    f"{stress_metrics['Accuracy']:.3f}",
    delta=f"{stress_metrics['Accuracy'] - clean_metrics['Accuracy']:.3f}"
)

col3.metric(
    "Clean F1",
    f"{clean_metrics['F1 Score']:.3f}"
)

col4.metric(
    "Stressed F1",
    f"{stress_metrics['F1 Score']:.3f}",
    delta=f"{stress_metrics['F1 Score'] - clean_metrics['F1 Score']:.3f}"
)

st.subheader("Performance Degradation")

degradation_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Balanced Accuracy",
        "AUROC"
    ],
    "Clean": [
        clean_metrics["Accuracy"],
        clean_metrics["Precision"],
        clean_metrics["Recall"],
        clean_metrics["F1 Score"],
        clean_metrics["Balanced Accuracy"],
        clean_metrics["AUROC"]
    ],
    "Stressed": [
        stress_metrics["Accuracy"],
        stress_metrics["Precision"],
        stress_metrics["Recall"],
        stress_metrics["F1 Score"],
        stress_metrics["Balanced Accuracy"],
        stress_metrics["AUROC"]
    ]
})

st.dataframe(
    degradation_df.round(4),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Metrics compare the same held-out test rows under clean and "
    "simulated stress conditions. Results are specific to this experiment."
)
st.header("02. Robustness sweeps (degradation curves)")

SWEEPS = {
    "Feature Drift": np.linspace(0, 3, 7),
    "Measurement Noise": np.linspace(0, 2, 7),
    "Missing Values": np.linspace(0, 0.5, 6)
}


def run_sweep(ds_name, fault, drift_feats, n_seeds):
    grid = SWEEPS[fault]
    out = []

    for name, m in models.items():

        for s in grid:

            vals = []

            seeds = range(
                1 if fault == "Feature Drift" else n_seeds
            )

            for sd in seeds:

                Xs = apply_components(
                    X_test,
                    {fault},
                    {fault: s},
                    sd,
                    REF_STD,
                    list(drift_feats)
                )

                vals.append(
                    roc_auc_score(
                        y_test,
                        m.predict_proba(Xs)[:, 1]
                    )
                )

            out.append({
                "Model": name,
                "Severity": s,
                "AUROC": np.mean(vals),
                "Std over seeds": np.std(vals)
            })

    return pd.DataFrame(out)


if st.button("Run robustness leaderboard (all faults, ~20-40 s)"):

    board = []
    curves = {}

    with st.spinner("Sweeping severities..."):

        for fault in SWEEPS:

            df = run_sweep(
                DS_NAME,
                fault,
                tuple(drift_features),
                3
            )

            curves[fault] = df

            for n in models:

                d = df[df.Model == n]

                clean_auc = d.AUROC.iloc[0]

                bp = breaking_point(
                    d.Severity.values,
                    d.AUROC.values,
                    clean_auc - 0.10
                )

                board.append({
                    "Fault": fault,
                    "Model": n,
                    "AUDC (mean AUROC over sweep)": degradation_auc(
                        d.Severity.values,
                        d.AUROC.values
                    ),
                    "Worst AUROC": d.AUROC.min(),
                    "Severity where AUROC drops by 0.10": (
                        bp if bp is not None else np.nan
                    )
                })

    st.session_state["curves"] = curves
    st.session_state["board"] = pd.DataFrame(board)
    st.session_state["curves_ds"] = DS_NAME


if (
    "curves" in st.session_state
    and st.session_state.get("curves_ds") == DS_NAME
):

    cols = st.columns(3)

    for col, (fault, df) in zip(
        cols,
        st.session_state["curves"].items()
    ):

        col.plotly_chart(
            px.line(
                df,
                x="Severity",
                y="AUROC",
                color="Model",
                markers=True,
                title=fault
            ),
            width="stretch"
        )

    board = st.session_state["board"]

    st.dataframe(
        board.style.format({
            c: "{:.3f}"
            for c in board.columns
            if c not in ("Fault", "Model")
        }),
        width="stretch"
    )

    worst = (
        board.groupby("Model")[
            "AUDC (mean AUROC over sweep)"
        ]
        .min()
        .sort_values(ascending=False)
    )

    st.write(
        "**Worst-case AUDC across faults (higher = more robust):**"
    )

    st.dataframe(
        worst.to_frame().style.format("{:.3f}"),
        width="stretch"
    )

    st.caption(
        "NaN in the last column = the model never lost 0.10 AUROC "
        "within the tested range."
    )


# ==========================================
# 03 RELIABILITY VERDICT
# ==========================================

st.divider()

st.header(
    "03. Reliability verdict (transparent rule, not a trained detector)"
)

p_c = proba_clean[selected_model_name]
p_s = proba_stress[selected_model_name]

drop, lo, hi, _ = paired_bootstrap(
    y_test,
    p_c,
    p_s,
    "AUROC"
)

if lo <= 0:
    risk = "No significant degradation"
    msg = st.success

elif lo < 0.03:
    risk = "Low"
    msg = st.success

elif lo < 0.10:
    risk = "Medium"
    msg = st.warning

else:
    risk = "High"
    msg = st.error

c1, c2, c3 = st.columns(3)

c1.metric("Verdict", risk)
c2.metric("AUROC drop", f"{drop:.3f}")
c3.metric("95% CI of drop", f"[{lo:.3f}, {hi:.3f}]")

msg(
    "Rule: classification uses the LOWER bound of the 95% paired-bootstrap "
    "CI of the AUROC drop (<=0: not significant, <0.03 Low, "
    "<0.10 Medium, otherwise High). Thresholds are conventions "
    "chosen by us; adjust them to your application's risk tolerance."
)

st.subheader(
    "Label-free warning signals (available without ground truth)"
)

conf_c = np.maximum(p_c, 1 - p_c).mean()
conf_s = np.maximum(p_s, 1 - p_s).mean()

l1, l2, l3 = st.columns(3)

l1.metric(
    "Mean confidence",
    f"{conf_s:.3f}",
    f"{conf_s - conf_c:+.3f} vs clean"
)

l2.metric(
    "Predicted-positive rate",
    f"{(p_s >= .5).mean():.3f}",
    f"{(p_s >= .5).mean() - (p_c >= .5).mean():+.3f} vs clean"
)

l3.metric(
    "Mean |P(y=1) shift|",
    f"{np.abs(p_s - p_c).mean():.3f}"
)


# ==========================================
# 04 CAUSE ATTRIBUTION
# ==========================================

st.divider()

st.header(
    "04. Cause attribution (interventional Shapley over injected faults)"
)

st.write(
    "Each fault subset is applied to the clean test set; the total "
    "AUROC drop is split among faults with exact Shapley values. "
    "Shares sum to the total drop by construction."
)

if st.button("Run attribution"):

    comps = [
        c for c in COMPONENT_ORDER
        if c in active
    ]

    if len(comps) < 2:

        st.warning(
            "Enable at least two faults to apportion blame between them."
        )

    else:

        with st.spinner("Evaluating all fault subsets..."):

            phi, total, table = shapley_fault_attribution(
                models[selected_model_name],
                X_test,
                y_test,
                comps,
                severities,
                REF_STD,
                drift_features,
                n_seeds=5
            )

        df = pd.DataFrame({
            "Fault": list(phi),
            "AUROC drop attributed": list(phi.values()),
            "Share of total": [
                v / total if total > 1e-9 else np.nan
                for v in phi.values()
            ]
        })

        st.dataframe(
            df.style.format({
                "AUROC drop attributed": "{:.4f}",
                "Share of total": "{:.1%}"
            }),
            width="stretch"
        )

        st.plotly_chart(
            px.bar(
                df,
                x="Fault",
                y="AUROC drop attributed",
                title=f"Total drop = {total:.3f}"
            ),
            width="stretch"
        )

        st.caption(
            "Interaction effects are shared out by the Shapley rule; "
            "this is attribution of injected faults in a simulation, "
            "where the truth is known."
        )


if "Feature Drift" in active and drift_features:

    st.subheader(
        "Causal check: restore the drifted features"
    )

    a0, a1, a2, rec = feature_restoration(
        models[selected_model_name],
        X_test,
        X_stress,
        y_test,
        drift_features
    )

    st.write(
        f"AUROC clean **{a0:.3f}** → "
        f"stressed **{a1:.3f}** → "
        f"after restoring {', '.join(drift_features)} **{a2:.3f}** "
        f"(recovers **{rec:.0%}** of the lost AUROC)."
    )

    st.caption(
        "If restoring a feature does not recover performance, "
        "drift in that feature was not the main cause, whatever "
        "the drift test says."
    )


# ==========================================
# 05 STATISTICAL DRIFT DETECTION
# ==========================================

st.divider()
st.header("05. Statistical drift detection")

ref_sample = X_train.sample(
    len(X_test),
    random_state=0
)

ud = univariate_drift(
    ref_sample,
    X_stress
)

st.dataframe(
    ud.style.format({
        "KS stat": "{:.3f}",
        "p-value": "{:.2e}",
        "p (BH-adjusted)": "{:.2e}",
        "Wasserstein (std units)": "{:.3f}",
        "Missing rate": "{:.2%}"
    }),
    width="stretch"
)

st.caption(
    "Per-feature KS test with Benjamini-Hochberg correction. "
    "A feature is flagged only if it is BOTH significant and has "
    "an effect size above 0.1 std."
)

with st.spinner("Classifier two-sample test..."):

    auc, imp = classifier_two_sample_test(
        ref_sample,
        X_stress
    )

    null_auc = null_drift_score(ref_sample)

d1, d2 = st.columns(2)

d1.metric(
    "Multivariate drift score (C2ST AUROC)",
    f"{auc:.3f}"
)

d2.metric(
    "Null calibration (reference vs reference)",
    f"{null_auc:.3f}"
)

st.plotly_chart(
    px.bar(
        imp.reset_index().rename(
            columns={
                "index": "Feature",
                0: "Importance"
            }
        ),
        x="Feature",
        y="Importance",
        title="Where the shift lives (C2ST feature importance)"
    ),
    width="stretch"
)

st.caption(
    "Drift is not degradation: compare this section to section 03. "
    "A feature can drift a lot without hurting the model, and the "
    "model can fail without large univariate drift."
)


# ==========================================
# 06 FAILURE INVESTIGATION
# ==========================================

st.divider()
st.header("06. Failure investigation v2")

model = models[selected_model_name]

breakdown, cats = paired_failure_breakdown(
    model,
    X_test,
    X_stress,
    y_test
)

st.dataframe(
    breakdown,
    width="stretch"
)

st.caption(
    "'Newly broken' rows are the real autopsy: they were correct "
    "on clean data and wrong under stress. 'Inherent' errors would "
    "exist without any fault."
)

error = (
    model.predict(X_stress) != y_test
).astype(int)

failed = X_stress.copy()

failed["True label"] = y_test
failed["Predicted"] = model.predict(X_stress)
failed["Category"] = cats
failed["Confidence"] = np.maximum(p_s, 1 - p_s)

st.metric(
    "Failure rate",
    f"{error.mean():.2%}"
)

if error.sum() > 0:

    st.subheader(
        "Which features separate failures from correct predictions?"
    )

    fe = feature_error_auc(
        X_stress,
        error,
        REF_MEDIAN
    )

    st.dataframe(
        fe.style.format({
            "AUROC (feature -> error)": "{:.3f}",
            "Association strength": "{:.3f}"
        }),
        width="stretch"
    )

    st.subheader(
        "Failure regions (depth-3 decision tree on the error indicator)"
    )

    st.code(
        error_rules(
            X_stress,
            error,
            REF_MEDIAN
        )
    )

    cp = confidence_profile(
        p_s,
        y_test
    )

    st.write(
        f"**Confidently wrong:** {cp['confident_errors']} of "
        f"{cp['errors']} errors "
        f"({cp['confident_error_share']:.1%}) were made with ≥90% "
        f"confidence. Mean confidence when correct "
        f"{cp['mean_conf_correct']:.2f} vs wrong "
        f"{cp['mean_conf_wrong']:.2f}."
    )

    st.dataframe(
        failed[
            failed.Category == "Newly broken by stress"
        ].head(25),
        width="stretch"
    )

    st.subheader(
        "Importance shift (permutation importance, AUROC scoring)"
    )

    with st.spinner("Permutation importance..."):

        pi_c = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=5,
            random_state=0
        ).importances_mean

        pi_s = permutation_importance(
            model,
            X_stress,
            y_test,
            scoring="roc_auc",
            n_repeats=5,
            random_state=0
        ).importances_mean

    pi = pd.DataFrame({
        "Feature": X_test.columns,
        "Clean": pi_c,
        "Stressed": pi_s
    }).melt(
        "Feature",
        var_name="Data",
        value_name="AUROC lost when shuffled"
    )

    st.plotly_chart(
        px.bar(
            pi,
            x="Feature",
            y="AUROC lost when shuffled",
            color="Data",
            barmode="group"
        ),
        width="stretch"
    )

    st.caption(
        "Correlated features can split credit. Neither permutation "
        "importance nor feature-error associations prove cause."
    )


# ==========================================
# 07 MODEL RECOMMENDATION
# ==========================================

st.divider()

st.header(
    "07. Model recommendation (statistically qualified)"
)

perf_df = (
    performance_df
    .pivot(
        index="Model",
        columns="Condition",
        values="AUROC"
    )
    .reset_index()
    .rename(columns={
        "Clean": "AUROC (clean)",
        "Stressed": "AUROC (stressed)"
    })
)
order = (
    perf_df.sort_values(
        "AUROC (stressed)",
        ascending=False
    ).reset_index(drop=True)
)

best = order.loc[0, "Model"]

verdicts = []

for other in order.Model[1:]:

    d, l, h, pwin = paired_bootstrap(
        y_test,
        proba_stress[best],
        proba_stress[other],
        "AUROC"
    )

    verdicts.append({
        "Compared": f"{best} vs {other}",
        "AUROC difference": d,
        "95% CI": f"[{l:.3f}, {h:.3f}]",
        "P(best > other)": pwin,
        "Conclusion": (
            "Clear winner"
            if l > 0
            else "Not distinguishable"
        )
    })

vdf = pd.DataFrame(verdicts)

st.dataframe(
    vdf.style.format({
        "AUROC difference": "{:.3f}",
        "P(best > other)": "{:.1%}"
    }),
    width="stretch"
)

if (vdf.Conclusion == "Clear winner").all():

    st.success(
        f"Recommended under '{scenario_label}': **{best}** "
        "(significantly better than all others on AUROC)."
    )

else:

    tied = [best] + [
        r["Compared"].split(" vs ")[1]
        for r in verdicts
        if r["Conclusion"] != "Clear winner"
    ]

    st.warning(
        "No statistically meaningful difference between: "
        + ", ".join(tied)
        + ". Decide on calibration, robustness sweeps (section 02), or cost."
    )

st.caption(
    "Valid only for this dataset, scenario, and severity. "
    "See the leaderboard in section 02 for cross-scenario evidence."
)


# ==========================================
# 11 INVESTIGATION REPORT
# ==========================================

st.divider()
st.header("11. Investigation report")

roc_c = roc_auc_score(y_test, p_c)
roc_s = roc_auc_score(y_test, p_s)

report = f"""MODEL AUTOPSY X - INVESTIGATION REPORT
=====================================
Model under investigation : {selected_model_name}
Scenario                  : {scenario_label}
Severities                : {', '.join(f'{k}={severities[k]}' for k in COMPONENT_ORDER if k in active) or 'none'}
Drifted features          : {', '.join(drift_features) if 'Feature Drift' in active else 'n/a'}
Random seed               : {SEED}
Test rows                 : {len(y_test)}

PERFORMANCE
AUROC clean -> stressed   : {roc_c:.3f} -> {roc_s:.3f}
AUROC drop (95% CI)       : {drop:.3f} [{lo:.3f}, {hi:.3f}]
Failure rate              : {error.mean():.2%} ({int(error.sum())} / {len(y_test)})
Verdict (rule-based)      : {risk}

DRIFT
Multivariate drift score  : {auc:.3f} (null calibration {null_auc:.3f})
Features flagged          : {', '.join(ud.loc[ud['Drift flagged'], 'Feature']) or 'none'}

FAILURE BREAKDOWN
{breakdown.to_string(index=False)}

LIMITATIONS
- Dataset may be uploaded or built-in; injected stress conditions are simulated.
- Fault attribution estimates effects of injected faults, not proven real-world causes.
- Verdict thresholds are conventions, not validated against real deployments.
- Drift tests detect distribution change, not performance loss; see AUROC drop.
- Feature-error associations are descriptive, not causal.
"""

st.text(report)

st.download_button(
    "⬇️ Download investigation report",
    data=report,
    file_name="model_autopsy_report.txt",
    mime="text/plain"
)

st.caption(
    "MODEL AUTOPSY X | AI Reliability Intelligence"
)


# ==========================================
# FEATURE 12: AUTOPSY INTELLIGENCE ENGINE
# ==========================================

flagged_drift_features = ud.loc[
    ud["Drift flagged"],
    "Feature"
].astype(str).tolist()

missing_rate = float(
    X_stress.isna().mean().mean()
)

stressed_metrics = metric_suite(
    y_test,
    p_s
)

if error.sum() > 0:

    confidence_results = confidence_profile(
        p_s,
        y_test
    )

    confident_errors = confidence_results[
        "confident_errors"
    ]

else:

    confident_errors = 0


# Collect actual audit evidence

autopsy_evidence = {

    "baseline_score": float(roc_c),

    "stressed_score": float(roc_s),

    "drift_features": flagged_drift_features,

    "missing_rate": missing_rate,

    "ece": float(stressed_metrics["ECE"]),

    "confident_errors": int(confident_errors),

    "scenario": scenario_label,

    "selected_model": selected_model_name,

    "auroc_drop": float(drop),

    "risk_verdict": risk,

    "test_rows": int(len(y_test))
}


# Generate Feature 12 investigation

render_autopsy_intelligence(
    autopsy_evidence
)
