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
from sklearn.model_selection import StratifiedKFold

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


# Domain-informed directions/magnitudes. Tune scales + threshold with
# `python tune_baseline_weights.py`; validate with `python validate_models.py`.
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


def baseline_score(
    z: np.ndarray, cols: list[str], weights: dict[str, float] | None = None
) -> np.ndarray:
    wmap = BASELINE_WEIGHTS if weights is None else weights
    w = np.array([wmap.get(c, 0.0) for c in cols], dtype=np.float64)
    if np.all(w == 0):
        raise ValueError("No overlapping weights for selected demographic columns.")
    return z @ w


def signed_unit_weights(cols: list[str], reference: dict[str, float]) -> dict[str, float]:
    """±1 weights matching the sign of `reference` for each column (for ablations / CV)."""
    out: dict[str, float] = {}
    for c in cols:
        v = float(reference.get(c, 1.0))
        if v > 0:
            out[c] = 1.0
        elif v < 0:
            out[c] = -1.0
        else:
            out[c] = 1.0
    return out


def baseline_used_columns(df: pd.DataFrame, reference: dict[str, float] | None = None) -> list[str]:
    """Ordered demographic columns that appear in `reference` with non-zero weight."""
    ref = BASELINE_WEIGHTS if reference is None else reference
    cols = demographic_feature_columns(df)
    return [c for c in cols if c in ref and ref[c] != 0]


def scaled_pair_weights(
    used: list[str],
    reference: dict[str, float],
    scale_pos: float,
    scale_neg: float,
) -> dict[str, float]:
    """
    Scale positive-reference weights by `scale_pos` and negative ones by `scale_neg`
    (reference values are already signed; this scales magnitude per sign group).
    """
    out: dict[str, float] = {}
    for c in used:
        v = float(reference[c])
        if v > 0:
            out[c] = v * scale_pos
        elif v < 0:
            out[c] = v * scale_neg
        else:
            out[c] = 0.0
    return out


def baseline_cv_scores(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int,
    weights: dict[str, float] | None,
    random_state: int,
    reference: dict[str, float] | None = None,
    top_quantile: float | None = None,
) -> dict[str, np.ndarray]:
    """Stratified CV: z-score and score threshold fit inside each train fold only."""
    tq = TOP_QUANTILE if top_quantile is None else float(top_quantile)
    used = baseline_used_columns(df, reference)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc, f1, prec, rec = [], [], [], []
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        mu, sigma = fit_zscore(tr, used)
        z_tr = zscore_matrix(tr, used, mu, sigma)
        z_va = zscore_matrix(va, used, mu, sigma)
        s_tr = baseline_score(z_tr, used, weights)
        s_va = baseline_score(z_va, used, weights)
        thresh = float(np.quantile(s_tr, tq))
        pred_va = (s_va >= thresh).astype(int)
        y_va = y[va_idx]
        acc.append(float(accuracy_score(y_va, pred_va)))
        f1.append(float(f1_score(y_va, pred_va, zero_division=0)))
        prec.append(float(precision_score(y_va, pred_va, zero_division=0)))
        rec.append(float(recall_score(y_va, pred_va, zero_division=0)))
    return {
        "accuracy": np.asarray(acc, dtype=np.float64),
        "f1": np.asarray(f1, dtype=np.float64),
        "precision": np.asarray(prec, dtype=np.float64),
        "recall": np.asarray(rec, dtype=np.float64),
    }


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

    used = baseline_used_columns(train)

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
