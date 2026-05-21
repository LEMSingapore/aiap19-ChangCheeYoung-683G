"""Central configuration for the AgroTech ML pipeline.

This module is the single source of truth for every tunable value in the
pipeline. It loads ``config.yaml`` from the repository root and exposes the
parsed contents as a module-level ``CONFIG`` dict plus a small number of
convenience constants.

Design intent
-------------
Nothing else in ``src/`` hard-codes a path, a column name, or a hyperparameter.
Every other module imports from here, so the pipeline is reconfigured by
editing ``config.yaml`` alone -- no code changes required. This satisfies the
assessment brief's requirement that the pipeline be "easily configurable to
enable easy experimentation of different algorithms and parameters".

The repository root is resolved by walking upward from this file until the
``data`` directory and ``config.yaml`` are both found, so the pipeline runs
correctly regardless of the working directory it is invoked from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ──────────────────────────────────────────────────────────────────────────
# Repository root resolution
# ──────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    """Locate the repository root independent of the current working directory.

    Walks upward from this file's location until a directory is found that
    contains both ``config.yaml`` and a ``data`` subdirectory -- the markers
    of the project root. Falls back to the parent of ``src/`` if no marker is
    found, which keeps a clear error surfacing downstream rather than here.

    Returns:
        Path: Absolute path to the repository root.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "config.yaml").is_file() and (candidate / "data").is_dir():
            return candidate
    # Fallback: src/ is one level below the root by project convention.
    return here.parent.parent


REPO_ROOT: Path = _find_repo_root()
CONFIG_PATH: Path = REPO_ROOT / "config.yaml"


# ──────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────
def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and parse the YAML configuration file.

    Args:
        path (Path | None): Path to the config file. Defaults to
            ``config.yaml`` at the repository root.

    Returns:
        dict: Parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = path or CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. It must sit at the repository root."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG: dict[str, Any] = load_config()


# ──────────────────────────────────────────────────────────────────────────
# Convenience constants — derived once from CONFIG so callers stay terse
# ──────────────────────────────────────────────────────────────────────────
# Paths are resolved to absolute form against the repo root so that callers
# never depend on their own working directory.
DB_PATH: Path = REPO_ROOT / CONFIG["data"]["db_path"]
TABLE_NAME: str = CONFIG["data"]["table_name"]
ARTIFACTS_DIR: Path = REPO_ROOT / CONFIG["data"]["artifacts_dir"]

TARGET_REGRESSION: str = CONFIG["targets"]["regression"]
TARGET_CLASSIFICATION: str = CONFIG["targets"]["classification"]

RANDOM_STATE: int = CONFIG["evaluation"]["random_state"]
TEST_SIZE: float = CONFIG["evaluation"]["test_size"]
CV_FOLDS: int = CONFIG["evaluation"]["cv_folds"]


def artifacts_dir() -> Path:
    """Return the artifacts directory, creating it if necessary.

    Returns:
        Path: Absolute path to the artifacts directory.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
