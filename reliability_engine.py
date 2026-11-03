import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_reliability_detector():

    np.random.seed(42)

    data = []

    # Generate synthetic reliability experiments
    for _ in range(5000):

        accuracy_drop = np.random.uniform(0, 0.60)
        f1_drop = np.random.uniform(0, 0.60)
        missing_rate = np.random.uniform(0, 0.50)
        noise_level = np.random.uniform(0, 2)
        drift_score = np.random.uniform(0, 1)

        risk_score = (
            accuracy_drop * 0.30
            + f1_drop * 0.30
            + missing_rate * 0.15
            + (noise_level / 2) * 0.10
            + drift_score * 0.15
        )

        if risk_score < 0.15:
            risk = "Low"
        elif risk_score < 0.30:
            risk = "Medium"
        else:
            risk = "High"

        data.append([
            accuracy_drop,
            f1_drop,
            missing_rate,
            noise_level,
            drift_score,
            risk
        ])

    df = pd.DataFrame(data, columns=[
        "accuracy_drop",
        "f1_drop",
        "missing_rate",
        "noise_level",
        "drift_score",
        "risk"
    ])

    X = df.drop("risk", axis=1)
    y = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    detector = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    detector.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        detector.predict(X_test)
    )

    return detector, accuracy


def diagnose_reliability(
    detector,
    accuracy_drop,
    f1_drop,
    missing_rate,
    noise_level,
    drift_score
):

    sample = pd.DataFrame([{
        "accuracy_drop": accuracy_drop,
        "f1_drop": f1_drop,
        "missing_rate": missing_rate,
        "noise_level": noise_level,
        "drift_score": drift_score
    }])

    prediction = detector.predict(sample)[0]
    probabilities = detector.predict_proba(sample)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence