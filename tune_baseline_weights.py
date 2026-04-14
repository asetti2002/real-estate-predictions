"""
Grid search for baseline score weights (positive vs negative feature groups) and
the train-set score quantile used as the classification threshold.

Search space (compact, interpretable):
  weight[c] = BASELINE_WEIGHTS[c] * scale_pos   if BASELINE_WEIGHTS[c] > 0
  weight[c] = BASELINE_WEIGHTS[c] * scale_neg   if BASELINE_WEIGHTS[c] < 0

Objective: maximize mean out-of-fold F1 (stratified CV on --train only).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from baseline_scoring import (
    BASELINE_WEIGHTS,
    TOP_QUANTILE,
    baseline_cv_scores,
    baseline_score,
    baseline_used_columns,
    fit_zscore,
    scaled_pair_weights,
    zscore_matrix,
)


def default_scale_grid() -> list[float]:
    return [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def default_quantile_grid() -> list[float]:
    return [0.65, 0.70, 0.75, 0.80, 0.85]


def main() -> None:
    p = argparse.ArgumentParser(description="Grid-search baseline weight scales + threshold quantile.")
    p.add_argument("--train", default=os.path.join("data", "output", "train.csv"))
    p.add_argument("--test", default=os.path.join("data", "output", "test.csv"))
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=None,
        help="Grid for scale_pos and scale_neg (default: built-in list).",
    )
    p.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=None,
        help="Grid for top_quantile (default: built-in list).",
    )
    p.add_argument(
        "--save-json",
        default=None,
        help="Write best settings + full weight dict to this path (JSON).",
    )
    p.add_argument(
        "--holdout",
        action="store_true",
        help="After CV, refit z-score on full train and report metrics on --test.",
    )
    args = p.parse_args()

    scales = args.scales if args.scales is not None else default_scale_grid()
    quantiles = args.quantiles if args.quantiles is not None else default_quantile_grid()

    train = pd.read_csv(args.train, low_memory=False)
    y = train["label"].to_numpy()
    used = baseline_used_columns(train, BASELINE_WEIGHTS)

    best: dict[str, float | dict[str, float]] | None = None

    for tq in quantiles:
        for sp in scales:
            for sn in scales:
                w = scaled_pair_weights(used, BASELINE_WEIGHTS, sp, sn)
                sc = baseline_cv_scores(
                    train,
                    y,
                    n_splits=args.cv_folds,
                    weights=w,
                    random_state=args.random_state,
                    top_quantile=tq,
                )
                mf = float(sc["f1"].mean())
                sf = float(sc["f1"].std())
                candidate = {
                    "mean_f1": mf,
                    "std_f1": sf,
                    "mean_precision": float(sc["precision"].mean()),
                    "mean_recall": float(sc["recall"].mean()),
                    "scale_pos": sp,
                    "scale_neg": sn,
                    "top_quantile": tq,
                    "weights": w,
                }
                if best is None or mf > float(best["mean_f1"]) or (
                    abs(mf - float(best["mean_f1"])) < 1e-9 and sf < float(best["std_f1"])
                ):
                    best = candidate

    assert best is not None

    w_best = best["weights"]
    assert isinstance(w_best, dict)

    if args.save_json:
        out = {
            "scale_pos": best["scale_pos"],
            "scale_neg": best["scale_neg"],
            "top_quantile": best["top_quantile"],
            "mean_f1": best["mean_f1"],
            "std_f1": best["std_f1"],
            "mean_precision": best["mean_precision"],
            "mean_recall": best["mean_recall"],
            "weights": w_best,
            "reference": BASELINE_WEIGHTS,
            "used_columns": used,
        }
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    if args.holdout and os.path.isfile(args.test):
        test = pd.read_csv(args.test, low_memory=False)
        y_te = test["label"].to_numpy()
        tq = float(best["top_quantile"])
        w = w_best
        mu, sigma = fit_zscore(train, used)
        z_tr = zscore_matrix(train, used, mu, sigma)
        z_te = zscore_matrix(test, used, mu, sigma)
        s_tr = baseline_score(z_tr, used, w)
        s_te = baseline_score(z_te, used, w)
        thresh = float(np.quantile(s_tr, tq))
        pred = (s_te >= thresh).astype(int)
        _ = (
            accuracy_score(y_te, pred),
            f1_score(y_te, pred, zero_division=0),
            precision_score(y_te, pred, zero_division=0),
            recall_score(y_te, pred, zero_division=0),
        )


if __name__ == "__main__":
    main()
