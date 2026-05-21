# AgroTech Innovations — Temperature & Plant Type-Stage ML Pipeline

AIAP Batch 19 Technical Assessment — End-to-end machine learning pipeline.

---

## a. Candidate

**Full name (as in NRIC):** Chang Chee Young
**Email address (as in application form):** cheeyoung.chang@gmail.com

---

## b. Overview of the submission

This repository contains an end-to-end machine learning pipeline for AgroTech
Innovations' controlled-environment farming data. It addresses the two
prediction tasks defined in the brief:

- **Regression** — predict `Temperature Sensor (°C)`, the temperature inside
  the closed environment.
- **Classification** — predict the composite `Plant Type-Stage` label, a
  twelve-class target combining plant type and growth stage.

The exploratory analysis behind every pipeline decision is in `eda.ipynb`. The
pipeline itself is implemented as Python modules under `src/` and is run
through `run.sh`.

### Folder structure

```
.
├── .github/                 GitHub Actions workflow (provided by template)
├── artifacts/               Pipeline outputs — trained models and metrics
│   ├── regression_metrics.json
│   └── classification_metrics.json
├── data/                    agri.db is placed here (not tracked by git)
├── src/
│   ├── settings.py          Config loader — resolves repo root, reads config.yaml
│   ├── preprocessing.py     SQLite loading and data cleaning
│   ├── feature_engineering.py  Task-specific feature selection and the
│   │                           preprocessing transformer
│   ├── models.py            Model construction, evaluation, persistence
│   └── main.py              Command-line entry point and orchestration
├── config.yaml              All tunable parameters
├── eda.ipynb                Exploratory Data Analysis (Task 1)
├── requirements.txt         Pinned dependencies
├── run.sh                   Executable pipeline runner
└── README.md
```

The trained model files (`artifacts/*.joblib`) are regenerated on every run
and are intentionally not tracked by git; the metrics JSON files are kept so
results are visible without running the pipeline.

---

## c. Running the pipeline

### Prerequisites

1. Place the dataset at `data/agri.db`. It is not tracked by git, per the
   brief. Download it from the assessment data URL into the `data/` folder.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Execution

```bash
bash run.sh
```

`run.sh` runs both tasks end to end and writes results to `artifacts/`. It does
not install dependencies — that is handled separately, as the brief specifies.
If a local virtual environment exists at `.venv/`, the script activates it; in
a clean CI environment that step is skipped and the pre-installed environment
is used.

To run a single task directly:

```bash
python src/main.py --task regression
python src/main.py --task classification
python src/main.py --task all          # default
```

### Changing parameters

All tunable values live in `config.yaml` — no code change is needed to
experiment. Models are swapped by editing the `models` section; preprocessing
behaviour, the test-set fraction, the cross-validation fold count, and the
random seed are all set there. `src/settings.py` reads this file and exposes
the values to the rest of the pipeline.

---

## d. Pipeline flow

The pipeline runs the same five stages for each task:

```
  agri.db (SQLite)
        │
        ▼
  load_data            Read the farm_data table
        │
        ▼
  clean_data           Normalise plant columns, coerce nutrient strings,
        │              repair sign-flipped sensors, drop duplicates,
        │              build the composite classification target
        ▼
  build_features       Select task-specific features (leakage guard),
        │              add the humidity indicator, assemble the
        │              ColumnTransformer
        ▼
  train_and_evaluate   Train baseline + XGBoost, 5-fold cross-validation,
        │              held-out test evaluation, select the best model
        ▼
  save_artifacts       Persist the best model and a metrics summary
```

Cleaning is task-agnostic and runs once per task identically. Feature selection
is task-specific — see section (g) on leakage. Preprocessing (imputation,
scaling, encoding) is the first step of a scikit-learn `Pipeline`, so it is
re-fitted on the training data of every cross-validation fold; the imputer and
scaler never see test data.

---

## e. Key EDA findings

Full detail is in `eda.ipynb`; the decisions that shaped the pipeline are:

- **The dataset has 7,489 full-row duplicates (≈13%).** With eight
  high-resolution sensors per row, exact coincidence at that rate is
  implausible — a synthetic-data artefact. They are dropped, leaving 50,000
  rows.
- **Three sensors carry physically-impossible negative readings.** Temperature
  (1,252 values), Light Intensity (1,385) and EC (14) all have sign-flipped
  entries. The EDA verified that every negative value, once passed through
  `abs()`, lands inside the legitimate positive range — confirming a sign-flip
  artefact rather than corrupt data — so the readings are repaired, not
  discarded.
- **The nutrient sensor columns are stored as strings**, some carrying a
  `' ppm'` suffix. They are coerced to numeric.
- **`Plant Type` and `Plant Stage` are inflated by casing/whitespace variants**
  — 12 and 9 raw values respectively, against a true 4 and 3. They are
  normalised before the composite target is built; without this the target
  would carry roughly 108 spurious classes.
- **Humidity is 67.6% missing.** Too sparse to rely on, but the missingness is
  judged informative (sensors offline, not missing at random), so a binary
  `humidity_present` indicator is engineered alongside median imputation.
- **`System Location Code` and `Previous Cycle Plant Type` are uninformative**
  — their distributions are uniform across all classes — and are dropped from
  both feature sets.
- **Temperature is bimodal**, with peaks around 22 °C and 24 °C reflecting
  per-plant-type setpoints. This favours tree-based models, which partition the
  modes, over linear models, which smear them — borne out in the results below.

---

## f. Feature processing

| Feature group | Columns | Processing |
|---|---|---|
| Numeric sensors | Light Intensity, CO2, EC, O2, pH, Water Level | Median imputation → standard scaling |
| Nutrient sensors | Nutrient N / P / K | String → numeric coercion → median imputation → standard scaling |
| Humidity | Humidity Sensor (%) | Median imputation → standard scaling; `humidity_present` indicator added |
| Plant categoricals | Plant Type, Plant Stage | Whitespace/casing normalisation; one-hot encoded — **regression only** (excluded from classification, see (g)) |
| Composite target | Plant Type-Stage | Constructed from normalised Plant Type + Plant Stage; label-encoded for the classifier |
| Dropped | System Location Code, Previous Cycle Plant Type | Removed from both tasks — uninformative |

Numeric imputation and scaling are applied inside the model `Pipeline`, fitted
per training fold, so no test-set statistics leak into training.

---

## g. Choice of models

Each task uses a simple baseline and a gradient-boosted main model. The
baseline establishes a reference score; a model is only worth its complexity if
it beats one.

**Regression — Ridge baseline, XGBoost main.**
Ridge is a regularised linear baseline; the L2 penalty steadies coefficients
given the multicollinear nutrient cluster found in the EDA. XGBoost is the main
model: temperature is bimodal and driven largely by plant type, and a
tree-based model partitions those modes naturally where a linear model cannot.

**Classification — Logistic Regression baseline, XGBoost main.**
Logistic Regression is an interpretable linear baseline. XGBoost is the main
model — it captures the non-linear sensor interactions that separate the twelve
classes.

**Leakage guard.** The classification target `Plant Type-Stage` is constructed
from `Plant Type` and `Plant Stage`. Including either as a feature would let the
model reconstruct the target directly, so both are excluded from the
classification feature set. They remain valid features for the regression task,
where plant identity is known at prediction time and is not derived from
temperature. This task-dependent selection is enforced in
`feature_engineering.py`.

Hyperparameters are set deliberately in `config.yaml`, with the rationale for
each noted there. An exhaustive search (grid or Bayesian) is a documented next
step rather than part of this submission — see (i).

---

## h. Evaluation

Models are evaluated with 5-fold cross-validation on the training split, then
once on a held-out 20% test set. Cross-validation gives an honest estimate
without touching the test data; the held-out set is the final unbiased check.

### Regression — predicting Temperature (°C)

| Model | CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Ridge (baseline) | 1.152 | 0.889 | 1.130 | 0.505 |
| **XGBoost** | **0.971** | **0.746** | **0.955** | **0.646** |

XGBoost is selected. It predicts temperature to within roughly 0.75 °C on
average (MAE) and explains 65% of variance. The margin over the linear baseline
is the bimodality effect predicted in the EDA.

**Metrics.** MAE and RMSE are both in °C, so they are directly interpretable;
RMSE is reported alongside MAE because it penalises large errors more heavily.
R² gives the share of variance explained.

### Classification — predicting Plant Type-Stage (12 classes)

| Model | CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Logistic Regression (baseline) | 0.714 | 0.716 | 0.714 |
| **XGBoost** | **0.782** | **0.788** | **0.787** |

XGBoost is selected, at 0.79 macro-F1 against a roughly 0.08 random baseline
for twelve classes — and trained without the leaked plant columns, so the score
reflects genuine signal in the sensor data. Per-class F1 on the test set ranges
from about 0.48 (`Herbs — Vegetative`) to 1.00 (`Leafy Greens — Seedling`).

**Metrics.** The classes are near-balanced (imbalance ratio ≈ 1.07), so
accuracy is meaningful. Macro-F1 is reported as the primary metric because it
averages F1 across classes with equal weight, exposing weak per-class
performance that accuracy alone would hide. The full per-class precision/recall
report is saved in `artifacts/classification_metrics.json`.

---

## i. Other considerations for deployment

- **Hyperparameter search.** Hyperparameters are currently fixed, deliberate
  values. A grid or Bayesian search (e.g. Optuna) is the natural next step for
  additional performance.
- **Schema and drift validation.** The pipeline assumes the sensor schema is
  stable. Production deployment should validate incoming data against an
  expected schema and monitor for distribution drift, since the model is
  trained on a fixed snapshot.
- **Sign-flip handling.** Negative readings are repaired with `abs()` on the
  assumption they are sign-flipped. If the upstream sensor fault changes
  character, that assumption needs revisiting; the repair logic is isolated in
  `preprocessing.py` for that reason.
- **Humidity sensor coverage.** With 67.6% of humidity readings missing, the
  feature is weak. Improving sensor coverage would likely lift regression
  performance, given humidity's correlation with temperature.
- **Retraining cadence.** Controlled-environment setpoints and crop schedules
  change. The model should be retrained on a regular cadence, with the
  cross-validation and held-out scores tracked over time to catch degradation.
- **Serving.** Both models are persisted as scikit-learn `Pipeline` objects
  that bundle preprocessing with the estimator, so serving needs only the raw
  feature columns — no separate preprocessing step to keep in sync.
