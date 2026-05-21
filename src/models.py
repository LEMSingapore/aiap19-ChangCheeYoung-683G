"""Model construction, evaluation, and persistence for the AgroTech pipeline.

For each task this module trains two models -- a simple, interpretable
baseline and a gradient-boosted main model -- and evaluates them with the
"light" protocol agreed for this submission:

* k-fold cross-validation on the training split, for an honest performance
  estimate that does not touch the test data;
* a single evaluation on a held-out test set, reported for the chosen model.

Hyperparameters are read from ``config.yaml`` -- they are deliberately chosen
values, not library defaults. An exhaustive hyperparameter search is the
documented next step rather than part of this submission.

Leakage is avoided structurally: preprocessing is the first step of a
scikit-learn ``Pipeline``, so the imputer and scaler are fitted only on the
training fold inside every cross-validation split and on the training split
for the held-out evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from settings import CONFIG, CV_FOLDS, RANDOM_STATE, TEST_SIZE, artifacts_dir

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ModelResult:
    """Holds the evaluation outcome for one model.

    Attributes:
        name (str): Human-readable model name.
        cv_score (float): Mean cross-validation score on the training split.
        cv_std (float): Standard deviation of the cross-validation scores.
        test_metrics (dict): Metrics computed once on the held-out test set.
        estimator (Pipeline): The fitted pipeline (preprocessing + model).
    """

    name: str
    cv_score: float
    cv_std: float
    test_metrics: dict[str, Any] = field(default_factory=dict)
    estimator: Pipeline | None = None


# ──────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────
def _build_estimator(task: str, role: str) -> tuple[str, Any]:
    """Instantiate a model from the config by task and role.

    Args:
        task (str): ``"regression"`` or ``"classification"``.
        role (str): ``"baseline"`` or ``"main"``.

    Returns:
        tuple[str, Any]: The configured model name and the estimator instance.

    Raises:
        ValueError: If the configured model name is not recognised.
    """
    spec = CONFIG["models"][task][role]
    name, params = spec["name"], spec["params"]

    registry = {
        "ridge": Ridge,
        "logistic_regression": LogisticRegression,
        "xgboost": XGBRegressor if task == "regression" else XGBClassifier,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}' in config for {task}/{role}.")

    return name, registry[name](**params)


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R^2).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict[str, float]: Metric name to value.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """Compute classification metrics (accuracy, macro-F1, per-class report).

    Args:
        y_true (np.ndarray): True (encoded) labels.
        y_pred (np.ndarray): Predicted (encoded) labels.
        labels (list[str]): Original class names, indexed by encoded value.

    Returns:
        dict[str, Any]: Accuracy, macro-F1, and the full per-class report.
    """
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        ),
    }


# ──────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ──────────────────────────────────────────────────────────────────────────
def _evaluate_one(
    name: str,
    estimator: Any,
    transformer: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    labels: list[str] | None,
) -> ModelResult:
    """Cross-validate, then fit and test a single model.

    The model is wrapped with the preprocessing transformer in a ``Pipeline``
    so that preprocessing is re-fitted on the training data of every CV fold,
    preventing train/test leakage of imputation and scaling statistics.

    Args:
        name (str): Model name.
        estimator (Any): Unfitted estimator instance.
        transformer (ColumnTransformer): Unfitted preprocessing transformer.
        X_train, X_test (pd.DataFrame): Feature splits.
        y_train, y_test (np.ndarray): Target splits.
        task (str): ``"regression"`` or ``"classification"``.
        labels (list[str] | None): Class names for classification, else None.

    Returns:
        ModelResult: Populated result with CV score, test metrics, and the
        fitted pipeline.
    """
    pipe = Pipeline(steps=[("preprocess", transformer), ("model", estimator)])

    scoring = "neg_root_mean_squared_error" if task == "regression" else "f1_macro"
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=CV_FOLDS, scoring=scoring, n_jobs=-1
    )
    # neg_* scores are returned negated by sklearn convention; flip back.
    cv_scores = -cv_scores if task == "regression" else cv_scores

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if task == "regression":
        test_metrics = _regression_metrics(y_test, y_pred)
    else:
        test_metrics = _classification_metrics(y_test, y_pred, labels or [])

    result = ModelResult(
        name=name,
        cv_score=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        test_metrics=test_metrics,
        estimator=pipe,
    )
    logger.info(
        "  %-20s CV %s = %.4f (+/- %.4f)",
        name,
        "RMSE" if task == "regression" else "macro-F1",
        result.cv_score,
        result.cv_std,
    )
    return result


def _select_best(results: list[ModelResult], task: str) -> ModelResult:
    """Pick the better model by cross-validation score.

    For regression the CV score is RMSE (lower is better); for classification
    it is macro-F1 (higher is better).

    Args:
        results (list[ModelResult]): Evaluated models.
        task (str): ``"regression"`` or ``"classification"``.

    Returns:
        ModelResult: The selected model.
    """
    if task == "regression":
        return min(results, key=lambda r: r.cv_score)
    return max(results, key=lambda r: r.cv_score)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    transformer: ColumnTransformer,
    task: str,
) -> tuple[ModelResult, list[ModelResult], LabelEncoder | None]:
    """Train the baseline and main model for a task and select the best.

    Args:
        X (pd.DataFrame): Feature matrix from ``feature_engineering``.
        y (pd.Series): Target series.
        transformer (ColumnTransformer): Unfitted preprocessing transformer.
        task (str): ``"regression"`` or ``"classification"``.

    Returns:
        tuple: ``(best, all_results, label_encoder)``. ``label_encoder`` is
        ``None`` for regression.
    """
    label_encoder: LabelEncoder | None = None
    y_arr: np.ndarray = y.to_numpy()
    labels: list[str] | None = None

    # XGBoost's classifier requires integer-encoded targets.
    if task == "classification":
        label_encoder = LabelEncoder()
        y_arr = label_encoder.fit_transform(y)
        labels = list(label_encoder.classes_)

    stratify = y_arr if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    logger.info(
        "Split: %d train / %d test rows", len(X_train), len(X_test)
    )

    results: list[ModelResult] = []
    for role in ("baseline", "main"):
        name, estimator = _build_estimator(task, role)
        # A fresh clone of the transformer per model keeps the pipelines
        # independent (the same unfitted object is otherwise shared).
        from sklearn.base import clone

        result = _evaluate_one(
            name,
            estimator,
            clone(transformer),
            X_train,
            X_test,
            y_train,
            y_test,
            task,
            labels,
        )
        results.append(result)

    best = _select_best(results, task)
    logger.info("Best model for %s: %s", task, best.name)
    return best, results, label_encoder


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────
def save_artifacts(
    task: str,
    best: ModelResult,
    all_results: list[ModelResult],
    label_encoder: LabelEncoder | None,
) -> None:
    """Persist the best model and a metrics summary to the artifacts directory.

    Writes three files per task: the fitted model pipeline (``.joblib``), the
    label encoder if applicable (``.joblib``), and a JSON metrics summary
    covering every model evaluated.

    Args:
        task (str): ``"regression"`` or ``"classification"``.
        best (ModelResult): The selected model.
        all_results (list[ModelResult]): Every model evaluated, for the report.
        label_encoder (LabelEncoder | None): Encoder to persist, if any.
    """
    out: Path = artifacts_dir()

    model_path = out / f"{task}_best_model.joblib"
    joblib.dump(best.estimator, model_path)
    logger.info("Saved model -> %s", model_path)

    if label_encoder is not None:
        enc_path = out / f"{task}_label_encoder.joblib"
        joblib.dump(label_encoder, enc_path)
        logger.info("Saved label encoder -> %s", enc_path)

    summary = {
        "task": task,
        "best_model": best.name,
        "models": [
            {
                "name": r.name,
                "cv_score": r.cv_score,
                "cv_std": r.cv_std,
                "test_metrics": r.test_metrics,
            }
            for r in all_results
        ],
    }
    metrics_path = out / f"{task}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved metrics -> %s", metrics_path)
