from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_TRAIN = os.path.join("data", "output", "train.csv")
DEFAULT_TEST = os.path.join("data", "output", "test.csv")

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


def metrics_block(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> None:
    print(f"\n--- {name} ---")
    print(
        f"accuracy={accuracy_score(y_true, y_pred):.4f}  "
        f"precision={precision_score(y_true, y_pred, zero_division=0):.4f}  "
        f"recall={recall_score(y_true, y_pred, zero_division=0):.4f}  "
        f"f1={f1_score(y_true, y_pred, zero_division=0):.4f}"
    )
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            print(f"roc_auc={roc_auc_score(y_true, y_proba):.4f}")
        except ValueError:
            pass
    print("confusion_matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_true, y_pred))
    print("classification_report:")
    print(classification_report(y_true, y_pred, digits=4))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--test", default=DEFAULT_TEST)
    args = p.parse_args()

    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)
    feat_cols = ml_feature_columns(train)

    X_train = train[feat_cols].to_numpy(dtype=np.float64)
    X_test = test[feat_cols].to_numpy(dtype=np.float64)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    lr = Pipeline(
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
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    metrics_block("Logistic Regression (test)", y_test, lr_pred, lr_proba)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    metrics_block("Random Forest (test)", y_test, rf_pred, rf_proba)


if __name__ == "__main__":
    main()
