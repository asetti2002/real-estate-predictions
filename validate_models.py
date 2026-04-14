"""
Data-driven checks: stratified CV, dummy baselines, and hold-out metrics.

Use this to sanity-check that learned models beat chance and that the hand-tuned
baseline is not worse than simpler weighting; CV reduces reliance on a single split.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, cross_validate

from baseline_scoring import (
    BASELINE_WEIGHTS,
    TOP_QUANTILE,
    baseline_cv_scores,
    baseline_score,
    baseline_used_columns,
    fit_zscore,
    signed_unit_weights,
    zscore_matrix,
)
from supervised_models import (
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    binary_metrics_dict,
    make_logistic_regression,
    make_random_forest,
    ml_feature_columns,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-validate models and compare to dummies.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--parallel-cv",
        action="store_true",
        help="Use n_jobs=-1 in cross_validate (can fail in some sandboxes). Default is single-threaded CV.",
    )
    args = p.parse_args()
    cv_n_jobs = -1 if args.parallel_cv else 1

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module="sklearn")

    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # Dummy baselines (no features — "numbers from thin air" should not beat these by accident)
    for strat in ("most_frequent", "stratified"):
        dc = DummyClassifier(strategy=strat, random_state=args.random_state)
        cross_validate(
            dc,
            np.zeros((len(train), 1)),
            y_train,
            cv=StratifiedKFold(
                n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
            ),
            scoring=("accuracy", "f1", "precision", "recall"),
            n_jobs=cv_n_jobs,
        )

    used_cols = baseline_used_columns(train)
    unit_w = signed_unit_weights(used_cols, BASELINE_WEIGHTS)

    for _, w in (
        ("Baseline (hand weights)", None),
        ("Baseline (±1 same signs)", unit_w),
    ):
        baseline_cv_scores(
            train,
            y_train,
            n_splits=args.cv_folds,
            weights=w,
            random_state=args.random_state,
        )

    feat_cols = ml_feature_columns(train)
    X_train = train[feat_cols].to_numpy(dtype=np.float64)
    X_test = test[feat_cols].to_numpy(dtype=np.float64)

    cv_split = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
    )

    for _, est in (
        ("LogisticRegression", make_logistic_regression()),
        ("RandomForest", make_random_forest(n_jobs=1)),
    ):
        cross_validate(
            est,
            X_train,
            y_train,
            cv=cv_split,
            scoring=("accuracy", "f1", "precision", "recall", "roc_auc"),
            n_jobs=cv_n_jobs,
        )

    mu, sigma = fit_zscore(train, used_cols)
    z_tr = zscore_matrix(train, used_cols, mu, sigma)
    z_te = zscore_matrix(test, used_cols, mu, sigma)
    s_tr_hand = baseline_score(z_tr, used_cols, None)
    s_tr_unit = baseline_score(z_tr, used_cols, unit_w)
    thresh_hand = float(np.quantile(s_tr_hand, TOP_QUANTILE))
    thresh_unit = float(np.quantile(s_tr_unit, TOP_QUANTILE))
    pred_te_hand = (baseline_score(z_te, used_cols, None) >= thresh_hand).astype(int)
    pred_te_unit = (baseline_score(z_te, used_cols, unit_w) >= thresh_unit).astype(int)

    for pred in (pred_te_hand, pred_te_unit):
        binary_metrics_dict(y_test, pred, None)

    lr = make_logistic_regression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    binary_metrics_dict(y_test, lr_pred, lr_proba)

    rf = make_random_forest(n_jobs=1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    binary_metrics_dict(y_test, rf_pred, rf_proba)


if __name__ == "__main__":
    main()
