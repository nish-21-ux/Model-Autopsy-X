import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split

DATASETS = ["Synthetic (make_classification)", "Breast Cancer (real, sklearn)"]


def load_split(name):
    if name == DATASETS[1]:
        d = load_breast_cancer(as_frame=True)
        X, y = d.data.copy(), d.target.to_numpy()
        X.columns = [f"Feature_{i}" for i in range(1, X.shape[1] + 1)]
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X, y = make_classification(n_samples=5000, n_features=8, n_informative=5, n_redundant=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 9)])
    return train_test_split(X, y, test_size=0.3, random_state=42)