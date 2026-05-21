"""Data loading and cleaning for the AgroTech ML pipeline.

This module turns the raw ``agri.db`` SQLite table into a single clean
dataframe. Every cleaning decision here is traceable to a section of the
exploratory analysis in ``eda.ipynb``; section references appear in the
docstrings so the pipeline and the notebook stay in step.

Pipeline position
------------------
``load_data`` → ``clean_data`` produce one cleaned dataframe. Task-specific
feature selection (notably the leakage-driven dropping of ``Plant Type`` /
``Plant Stage`` for the classification task) happens downstream in
``feature_engineering.py``, not here -- this module is task-agnostic.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from settings import CONFIG, DB_PATH, TABLE_NAME, TARGET_CLASSIFICATION

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────
def load_data(db_path: Path | None = None, table: str | None = None) -> pd.DataFrame:
    """Load the full sensor table from the SQLite database.

    Args:
        db_path (Path | None): Path to ``agri.db``. Defaults to the configured
            location.
        table (str | None): Table name. Defaults to the configured table.

    Returns:
        pd.DataFrame: Raw, unmodified contents of the table.

    Raises:
        FileNotFoundError: If the database file is absent. The brief states the
            database is not committed to the repository, so a missing file
            usually means it has not been downloaded into ``data/``.
    """
    db_path = db_path or DB_PATH
    table = table or TABLE_NAME

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Download agri.db into the data/ "
            f"folder (it is intentionally not tracked by git)."
        )

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)

    logger.info("Loaded %s: %d rows x %d columns", table, df.shape[0], df.shape[1])
    return df


# ──────────────────────────────────────────────────────────────────────────
# Individual cleaning steps (EDA §4)
# ──────────────────────────────────────────────────────────────────────────
def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact full-row duplicates (EDA §2).

    The EDA found 7,489 full-row duplicates (~13% of rows). With eight
    high-resolution numeric sensors per row, exact coincidence at that rate is
    statistically implausible -- these are a synthetic-data-generation artefact
    and are removed.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Dataframe with duplicate rows removed and the index reset.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows (%d remain)", n_before - len(df), len(df))
    return df


def coerce_nutrient_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the nutrient sensor columns from string to numeric (EDA §4c).

    ``Nutrient N / P / K Sensor (ppm)`` are stored as strings -- some values
    carry a ``' ppm'`` suffix. ``pd.to_numeric(errors='coerce')`` casts them to
    float; unparseable entries become ``NaN`` and are imputed later in the
    feature pipeline.

    Args:
        df (pd.DataFrame): Dataframe with raw nutrient columns.

    Returns:
        pd.DataFrame: Dataframe with nutrient columns as numeric dtype.
    """
    df = df.copy()
    for col in CONFIG["columns"]["nutrient_cols"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Coerced nutrient columns to numeric")
    return df


def repair_sign_flipped_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Repair sign-flipped sensor readings via abs() (EDA §4c, §4a-check).

    Temperature, Light Intensity, and EC contain physically-impossible negative
    values. The EDA's §4a-check cell verified empirically that every negative
    reading, once passed through ``abs()``, lands inside the legitimate positive
    range -- confirming a sign-flip artefact rather than corrupt data. Repairing
    with ``abs()`` recovers those rows instead of discarding them.

    Args:
        df (pd.DataFrame): Dataframe with possibly sign-flipped sensor columns.

    Returns:
        pd.DataFrame: Dataframe with the affected columns made non-negative.
    """
    df = df.copy()
    for col in CONFIG["columns"]["sign_flip_cols"]:
        n_neg = int((df[col] < 0).sum())
        df[col] = df[col].abs()
        if n_neg:
            logger.info("Sign-flip repair: %s — %d values", col, n_neg)
    return df


def normalise_plant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise casing/whitespace in the plant categorical columns (EDA §4b).

    Raw ``Plant Type`` has 12 unique values and ``Plant Stage`` has 9 -- both
    inflated by casing and whitespace variants. ``strip().title()`` collapses
    them to the true 4 types and 3 stages. Without this, the composite target
    would carry ~108 spurious classes instead of 12.

    Args:
        df (pd.DataFrame): Dataframe with raw plant columns.

    Returns:
        pd.DataFrame: Dataframe with normalised plant columns.
    """
    df = df.copy()
    for col in CONFIG["columns"]["plant_cols"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df


def build_composite_target(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the composite ``Plant Type-Stage`` classification target.

    The brief defines the classification target as the combined Plant
    Type-Stage label. It is built by concatenating the (already normalised)
    ``Plant Type`` and ``Plant Stage`` columns.

    Args:
        df (pd.DataFrame): Dataframe with normalised plant columns.

    Returns:
        pd.DataFrame: Dataframe with the composite target column added.
    """
    df = df.copy()
    df[TARGET_CLASSIFICATION] = df["Plant Type"] + " — " + df["Plant Stage"]
    logger.info(
        "Built composite target '%s' (%d classes)",
        TARGET_CLASSIFICATION,
        df[TARGET_CLASSIFICATION].nunique(),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps in the correct order.

    Order matters: plant columns are normalised *before* the composite target
    is built from them, and *before* duplicates are dropped (so casing-only
    variants collapse consistently).

    Args:
        df (pd.DataFrame): Raw dataframe from :func:`load_data`.

    Returns:
        pd.DataFrame: Fully cleaned, task-agnostic dataframe.
    """
    df = normalise_plant_columns(df)
    df = coerce_nutrient_columns(df)
    df = repair_sign_flipped_sensors(df)
    if CONFIG["preprocessing"]["drop_duplicates"]:
        df = drop_duplicate_rows(df)
    df = build_composite_target(df)
    return df


def get_clean_data() -> pd.DataFrame:
    """Convenience entry point: load from SQLite and return cleaned data.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for feature engineering.
    """
    return clean_data(load_data())
