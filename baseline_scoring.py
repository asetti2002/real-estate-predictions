from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ZHVI_DERIVED = {
    "current_zhvi",
    "return_3m",
    "return_12m",
    "return_36m",
    "return_60m",
    "acceleration",
    "volatility_12m",
    "vs_metro",
}


# might do a grid search to find the best weights
BASELINE_WEIGHTS: dict[str, float] = {
    "median_household_income": 1.0,
    "employment_rate": 1.0,
    "bachelors_degree_pct": 1.0,
    "graduate_degree_pct": 1.0,
    "pct_18_to_34": 1.0,
    "owner_occupied_pct": 0.8,
    "health_insurance_pct": 0.5,
    "total_population": 0.3,
    "total_housing_units": 0.2,
    "total_households": 0.2,
    "median_gross_rent": 0.3,
    "median_home_value": 0.2,
    "median_age": -0.8,
    "pct_65_plus": -1.0,
    "vacancy_rate": -1.0,
}

DEFAULT_TRAIN = os.path.join("data", "output", "train.csv")
DEFAULT_TEST = os.path.join("data", "output", "test.csv")
DEFAULT_PRED_OUT = os.path.join("data", "output", "baseline_predictions.csv")
TOP_QUANTILE = 0.75


def demographic_feature_columns(df: pd.DataFrame) -> list[str]:
    skip = ZHVI_DERIVED | {"growth_forecast_1yr", "label", "RegionName"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def fit_zscore(train: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mu = train[cols].mean(axis=0).to_numpy(dtype=np.float64)
    sigma = train[cols].std(axis=0).replace(0, np.nan).to_numpy(dtype=np.float64)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    return mu, sigma


def zscore_matrix(df: pd.DataFrame, cols: list[str], mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float64)
    return (x - mu) / sigma


def baseline_score(z: np.ndarray, cols: list[str]) -> np.ndarray:
    w = np.array([BASELINE_WEIGHTS.get(c, 0.0) for c in cols], dtype=np.float64)
    if np.all(w == 0):
        raise ValueError("No overlapping weights for selected demographic columns.")
    return z @ w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument(
        "--save-preds",
        nargs="?",
        const=DEFAULT_PRED_OUT,
        default=None,
    )
    args = p.parse_args()

    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)

    cols = demographic_feature_columns(train)
    used = [c for c in cols if c in BASELINE_WEIGHTS and BASELINE_WEIGHTS[c] != 0]

    mu, sigma = fit_zscore(train, used)
    z_train = zscore_matrix(train, used, mu, sigma)
    z_test = zscore_matrix(test, used, mu, sigma)

    s_train = baseline_score(z_train, used)
    s_test = baseline_score(z_test, used)

    thresh = float(np.quantile(s_train, TOP_QUANTILE))
    pred_train = (s_train >= thresh).astype(int)
    pred_test = (s_test >= thresh).astype(int)

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    print(f"\nScore threshold (train {TOP_QUANTILE:.0%} quantile): {thresh:.4f}")
    print("\n--- Train (calibration) ---")
    print(
        f"accuracy={accuracy_score(y_train, pred_train):.4f}  "
        f"precision={precision_score(y_train, pred_train, zero_division=0):.4f}  "
        f"recall={recall_score(y_train, pred_train, zero_division=0):.4f}  "
        f"f1={f1_score(y_train, pred_train, zero_division=0):.4f}"
    )
    print("confusion_matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_train, pred_train))

    print("\n--- Test ---")
    print(
        f"accuracy={accuracy_score(y_test, pred_test):.4f}  "
        f"precision={precision_score(y_test, pred_test, zero_division=0):.4f}  "
        f"recall={recall_score(y_test, pred_test, zero_division=0):.4f}  "
        f"f1={f1_score(y_test, pred_test, zero_division=0):.4f}"
    )
    print("confusion_matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, pred_test))
    print("\nclassification_report (test):")
    print(classification_report(y_test, pred_test, digits=4))

    if args.save_preds:
        out = test[
            [c for c in ["RegionName", "State", "Metro"] if c in test.columns]
        ].copy()
        out["baseline_score"] = s_test
        out["pred_high_growth"] = pred_test
        out["label"] = y_test
        os.makedirs(os.path.dirname(args.save_preds) or ".", exist_ok=True)
        out.to_csv(args.save_preds, index=False)
        print(f"\nWrote: {args.save_preds}")


if __name__ == "__main__":
    main()
