
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline


def prepare_uploaded_dataset(df, target_column):
    """Validate, split and preprocess an uploaded binary classification CSV."""

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Your CSV contains duplicate column names.")

    if target_column not in df.columns:
        raise ValueError("Please select a valid target column.")

    if df[target_column].isna().any():
        raise ValueError("The target column contains missing values.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # Remove columns containing no usable values.
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No usable feature columns were found.")

    # Reject datetime columns for this initial numerical pipeline.
    datetime_cols = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    if datetime_cols:
        raise ValueError(
            "Please convert datetime columns into numerical features "
            "before uploading: " + ", ".join(datetime_cols)
        )

    # Convert infinite numerical values into missing values.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[numeric_cols] = X[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    # Convert categorical values to consistent strings.
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    for col in categorical_cols:
        X[col] = X[col].astype("string")

    # This version supports binary classification only.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    if len(encoder.classes_) != 2:
        raise ValueError(
            "Currently, Model Autopsy X supports binary classification "
            "datasets with exactly two target classes."
        )

    counts = np.bincount(y)

    if len(counts) != 2 or counts.min() < 2:
        raise ValueError(
            "Each target class needs at least two rows for a stratified split."
        )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        transformers.append(
            ("num", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        transformers.append(
            ("cat", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(
        transformers,
        verbose_feature_names_out=True
    )

    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train_raw.index
    )

    X_test = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test_raw.index
    )

    metadata = {
        "original_rows": len(df),
        "original_features": len(X.columns),
        "processed_features": len(feature_names),
        "target_classes": encoder.classes_.tolist(),
        "preprocessor": preprocessor
    }

    return X_train, X_test, y_train, y_test, metadata
