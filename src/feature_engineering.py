"""Feature engineering for the AgroTech ML pipeline.

This module turns the cleaned dataframe from ``preprocessing.py`` into a
model-ready feature matrix and target vector. It does two jobs:

1. **Task-specific feature selection.** The regression and classification
   tasks do not use the same columns. Most importantly, ``Plant Type`` and
   ``Plant Stage`` are *valid features for the regression task* (plant identity
   is known at prediction time) but *forbidden for the classification task* --
   the classification target is built from those two columns, so including
   them would be target leakage. This module is the single place that
   decision is enforced.

2. **The preprocessing transformer.** A scikit-learn ``ColumnTransformer``
   imputes, scales, and one-hot encodes. It is returned *unfitted* so that the
   caller can fit it inside a ``Pipeline`` on the training split only -- this
   avoids train/test leakage of imputation statistics and scaling parameters.

Pipeline position
------------------
``preprocessing.get_clean_data()`` → ``build_features(df, task)`` →
``(X, y, transformer)`` consumed by ``models.py``.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from settings import CONFIG, TARGET_CLASSIFICATION, TARGET_REGRESSION

logger = logging.getLogger(__name__)

# The eight raw numeric sensor columns plus the three nutrient columns. The
# regression target (Temperature) is handled as a label, never a feature.
_NUTRIENT_COLS = CONFIG["columns"]["nutrient_cols"]
_PLANT_COLS = CONFIG["columns"]["plant_cols"]
_DROP_COLS = CONFIG["columns"]["drop_cols"]
_HUMIDITY_COL = "Humidity Sensor (%)"
_HUMIDITY_INDICATOR = "humidity_present"


# ──────────────────────────────────────────────────────────────────────────
# Optional engineered feature
# ──────────────────────────────────────────────────────────────────────────
def add_humidity_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary ``humidity_present`` feature (EDA §3).

    Humidity is 67.6% missing, and the EDA judged the missingness to be
    informative (MAR/MNAR -- sensors offline rather than missing at random).
    A binary indicator lets the model learn from the *absence* of a reading,
    which is cheap signal that median imputation alone would discard.

    Args:
        df (pd.DataFrame): Cleaned dataframe.

    Returns:
        pd.DataFrame: Dataframe with the indicator column added.
    """
    df = df.copy()
    df[_HUMIDITY_INDICATOR] = df[_HUMIDITY_COL].notna().astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Task-specific feature selection
# ──────────────────────────────────────────────────────────────────────────
def select_features(df: pd.DataFrame, task: str) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical feature columns for a given task.

    Leakage rule
    ------------
    For the **classification** task the target ``Plant Type-Stage`` is derived
    from ``Plant Type`` and ``Plant Stage``; both are therefore excluded from
    the feature set. For the **regression** task they are legitimate predictors
    and are kept as categorical features.

    ``System Location Code`` and ``Previous Cycle Plant Type`` are dropped for
    both tasks -- the EDA (§8b) found them uniform across all classes and
    uninformative.

    Args:
        df (pd.DataFrame): Cleaned dataframe (after the humidity indicator is
            added).
        task (str): Either ``"regression"`` or ``"classification"``.

    Returns:
        tuple[list[str], list[str]]: ``(numeric_features, categorical_features)``.

    Raises:
        ValueError: If ``task`` is not a recognised value.
    """
    if task not in {"regression", "classification"}:
        raise ValueError(f"Unknown task '{task}'. Use 'regression' or 'classification'.")

    # Columns that are never features: the two targets and the dropped nuisance
    # columns. The composite target is also excluded.
    never_features = {
        TARGET_REGRESSION,
        TARGET_CLASSIFICATION,
        *_DROP_COLS,
    }
    # Plant Type / Plant Stage: features for regression, leakage for classification.
    if task == "classification":
        never_features.update(_PLANT_COLS)

    numeric_features = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in never_features
    ]
    categorical_features = [
        c
        for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in never_features
    ]

    logger.info(
        "Task '%s': %d numeric + %d categorical features",
        task,
        len(numeric_features),
        len(categorical_features),
    )
    return numeric_features, categorical_features


# ──────────────────────────────────────────────────────────────────────────
# Preprocessing transformer
# ──────────────────────────────────────────────────────────────────────────
def build_transformer(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build the unfitted preprocessing ``ColumnTransformer``.

    Numeric columns are median-imputed then standardised. Categorical columns
    are one-hot encoded (unknown categories at predict time are ignored rather
    than raising). The transformer is returned **unfitted**: the caller fits it
    on the training split only, so imputation medians and scaling statistics
    never see the test data.

    Args:
        numeric_features (list[str]): Numeric column names.
        categorical_features (list[str]): Categorical column names.

    Returns:
        ColumnTransformer: Unfitted preprocessing transformer.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame, task: str
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Produce the feature matrix, target, and transformer for a task.

    For the regression task, rows with a missing temperature target are
    dropped -- the target cannot be imputed without fabricating labels.

    Args:
        df (pd.DataFrame): Cleaned dataframe from ``preprocessing``.
        task (str): ``"regression"`` or ``"classification"``.

    Returns:
        tuple: ``(X, y, transformer)`` where ``X`` is the unprocessed feature
        frame, ``y`` is the target series, and ``transformer`` is the unfitted
        ``ColumnTransformer`` to be fitted inside the model pipeline.

    Raises:
        ValueError: If ``task`` is not recognised.
    """
    if task not in {"regression", "classification"}:
        raise ValueError(f"Unknown task '{task}'. Use 'regression' or 'classification'.")

    if CONFIG["preprocessing"]["add_humidity_indicator"]:
        df = add_humidity_indicator(df)

    target = TARGET_REGRESSION if task == "regression" else TARGET_CLASSIFICATION

    if task == "regression":
        n_before = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        logger.info(
            "Regression: dropped %d rows with missing target (%d remain)",
            n_before - len(df),
            len(df),
        )

    numeric_features, categorical_features = select_features(df, task)
    X = df[numeric_features + categorical_features]
    y = df[target]

    transformer = build_transformer(numeric_features, categorical_features)
    return X, y, transformer
