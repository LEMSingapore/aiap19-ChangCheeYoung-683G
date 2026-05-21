#!/usr/bin/env bash
# ============================================================================
# run.sh — execute the AgroTech end-to-end ML pipeline.
#
# Per the assessment brief, this script does NOT install dependencies. In the
# grading environment the dependencies from requirements.txt are installed
# beforehand; this script only runs the pipeline.
#
# For local development a project virtual environment (.venv) is activated if
# one is present. In CI there is no .venv, so that step is simply skipped and
# the pre-installed environment is used.
# ============================================================================

set -euo pipefail

# Resolve the directory this script lives in, so the pipeline runs correctly
# regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate local virtual environment if present (developer convenience) ───
if [ -f ".venv/bin/activate" ]; then
    echo "[run.sh] Activating local .venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "[run.sh] No .venv found — using the current environment"
fi

# ── Dependency guard: fail early with a clear message, not a deep traceback ─
python -c "import pandas, sklearn, xgboost, yaml" 2>/dev/null || {
    echo "[run.sh] ERROR: required packages are missing." >&2
    echo "[run.sh] Install them first:  pip install -r requirements.txt" >&2
    exit 1
}

# ── Run the pipeline ────────────────────────────────────────────────────────
echo "[run.sh] Starting pipeline"
python src/main.py --task all
echo "[run.sh] Done"
