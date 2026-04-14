from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN = os.path.join("data", "output", "train.csv")
TEST = os.path.join("data", "output", "test.csv")




ID_GEO_DROP = {
    "RegionName",
    "State",
    "StateName",
    "City",
    "Metro",
    "CountyName",
    "label",
    "growth_forecast_1yr",
}
def ml_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in ID_GEO_DROP:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def binary_metrics_dict(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            pass
    return out


def make_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
def make_random_forest(*, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=n_jobs,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=TRAIN)
    p.add_argument("--test", default=TEST)
    args = p.parse_args()

    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)
    feat_cols = ml_feature_columns(train)

    X_train = train[feat_cols].to_numpy(dtype=np.float64)
    X_test = test[feat_cols].to_numpy(dtype=np.float64)
    y_train = train["label"].to_numpy()



    lr = make_logistic_regression()
    lr.fit(X_train, y_train)
    lr.predict(X_test)
    lr.predict_proba(X_test)




    rf = make_random_forest()
    rf.fit(X_train, y_train)
    rf.predict(X_test)
    rf.predict_proba(X_test)


if __name__ == "__main__":
    main()
