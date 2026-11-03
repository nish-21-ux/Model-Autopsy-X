
import pandas as pd
import numpy as np


def investigate_failures(model, X_test, y_test):

    predictions = model.predict(X_test)

    failures = X_test.copy()

    failures["Actual"] = np.asarray(y_test)
    failures["Predicted"] = predictions

    failures["Failure"] = (
        failures["Actual"] != failures["Predicted"]
    )

    failed_cases = failures[
        failures["Failure"]
    ].copy()

    return failed_cases


def analyze_feature_patterns(failed_cases):

    if failed_cases.empty:
        return pd.DataFrame()

    feature_columns = [
        col for col in failed_cases.columns
        if col.startswith("Feature_")
    ]

    analysis = pd.DataFrame({
        "Feature": feature_columns,
        "Average Value in Failed Cases": [
            failed_cases[col].mean()
            for col in feature_columns
        ]
    })

    analysis["Absolute Impact Indicator"] = (
        analysis["Average Value in Failed Cases"].abs()
    )

    return analysis.sort_values(
        "Absolute Impact Indicator",
        ascending=False
    )
