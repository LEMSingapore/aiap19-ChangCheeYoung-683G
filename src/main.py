"""Pipeline entry point for the AgroTech ML pipeline.

Run from the command line, selecting one or both prediction tasks::

    python src/main.py --task regression
    python src/main.py --task classification
    python src/main.py --task all          # default

This is the orchestration layer: it wires together data loading, cleaning,
feature engineering, and (once implemented) model training and evaluation.
Configuration lives in ``config.yaml``; this script only chooses *which* task
to run.
"""

from __future__ import annotations

import argparse
import logging
import sys

from feature_engineering import build_features
from models import save_artifacts, train_and_evaluate, tune_xgboost
from preprocessing import get_clean_data

TASKS = ("regression", "classification")


def setup_logging() -> None:
    """Configure root logging to stdout with a plain, readable format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (list[str] | None): Argument list. Defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed arguments with a ``task`` attribute.
    """
    parser = argparse.ArgumentParser(
        description="AgroTech ML pipeline — temperature regression and "
        "Plant Type-Stage classification."
    )
    parser.add_argument(
        "--task",
        choices=(*TASKS, "all"),
        default="all",
        help="Which prediction task to run (default: all).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a RandomizedSearchCV hyperparameter search for the XGBoost "
        "model(s) instead of the standard fixed-config run. Off by default; "
        "run.sh and CI use the fast fixed-config path.",
    )
    return parser.parse_args(argv)


def run_task(task: str, tune: bool = False) -> None:
    """Run the pipeline for a single task.

    Loads and cleans the data and builds task-specific features. With
    ``tune=False`` (the default) it trains and evaluates a baseline and a
    gradient-boosted model and persists the best one. With ``tune=True`` it
    instead runs a RandomizedSearchCV hyperparameter search for the XGBoost
    model and saves a tuning report -- the standard train/evaluate path is
    skipped.

    Args:
        task (str): Either ``"regression"`` or ``"classification"``.
        tune (bool): If True, run the hyperparameter search instead of the
            standard pipeline.
    """
    log = logging.getLogger(__name__)
    log.info("=" * 60)
    log.info("TASK: %s%s", task, "  [TUNE]" if tune else "")
    log.info("=" * 60)

    df = get_clean_data()
    X, y, transformer = build_features(df, task)
    log.info("Feature matrix : %d rows x %d columns", X.shape[0], X.shape[1])

    if tune:
        tune_xgboost(X, y, transformer, task)
    else:
        best, all_results, label_encoder = train_and_evaluate(X, y, transformer, task)
        save_artifacts(task, best, all_results, label_encoder)

    log.info("Task '%s' complete.", task)


def main(argv: list[str] | None = None) -> int:
    """Program entry point.

    Args:
        argv (list[str] | None): Argument list. Defaults to ``sys.argv``.

    Returns:
        int: Process exit code (0 on success).
    """
    setup_logging()
    args = parse_args(argv)

    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        run_task(task, tune=args.tune)

    logging.getLogger(__name__).info("Pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
