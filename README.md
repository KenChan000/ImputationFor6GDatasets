# Benchmarking Tabular Imputation Methods for 6G Wireless Datasets

Code for the CSE3000 Research Project. This repository benchmarks seven tabular
imputation methods across four missingness mechanisms and three missing rates on
two DeepSense 6G scenarios, evaluated on reconstruction error, distributional
fidelity, and downstream beam-prediction accuracy.

It includes a bootstrap script that builds a Python virtual environment with all
the imputation libraries used in the project. Tested on Linux, macOS, and Windows
via **WSL** (recommended).

## Imputation methods covered

The environment supports the full method shortlist:

- **Mean** — `scikit-learn` (`SimpleImputer`)
- **kNN** — `scikit-learn` (`KNNImputer`)
- **MICE** — `scikit-learn` (`IterativeImputer`)
- **SoftImpute** — `fancyimpute.SoftImpute` (low-rank, important for CSI data)
- **HyperImpute** — `hyperimpute` package
- **GRAPE** — bundled inside DiffPuter at `external/DiffPuter/baselines/GRAPE/`
- **DiffPuter** — included as a git submodule at `external/DiffPuter/`

DiffPuter is a git submodule (not pip-installed) because it's a research codebase
rather than a packaged library. The submodule is **pinned to a specific commit**
via `.gitmodules`, so everyone builds against the same source.

## Repository layout

```
src/common/        shared experiment code (amputation, metrics, datasets, tuning, ...)
src/imputers/      imputer wrappers (imputers.py, grape_imputer.py, diffputer_imputer.py)
notebooks/         tuning.ipynb, imputation.ipynb, downstream.ipynb
results/csv/       result tables the paper is built from
results/figures/   generated figures
results/           cached imputations (.npy) and tuned_params.json
scripts/           setup_imputation_env.sh
requirements/      base.txt, pyg.txt, dev.txt
external/DiffPuter DiffPuter submodule (contains GRAPE under baselines/GRAPE/)
data/              DeepSense 6G scenarios (not redistributed — see Dataset below)
```

## Prerequisites

- Python 3.10–3.12 (with `venv` and `pip`)
- git
- A C/C++ compiler (`build-essential` on Debian/Ubuntu) — only needed if a
  prebuilt `torch-scatter` wheel isn't available for your torch+CUDA combo

## Dataset

The DeepSense 6G V2I scenarios (Scenario 5 and Scenario 33) are **not
redistributed** in this repository (non-commercial academic licence). Obtain them
directly from <https://deepsense6g.net> and place them under `data/` following the
structure expected by `src/common/deepsense_data.py`.

## Quick start

Clone **with submodules** so DiffPuter is populated:

```bash
git clone --recurse-submodules https://github.com/KenChan000/ImputationFor6GDatasets.git
cd ImputationFor6GDatasets
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

Then build the environment (the script anchors to the repo root, so it works from
anywhere; it also initialises the submodule if needed):

```bash
chmod +x scripts/setup_imputation_env.sh
./scripts/setup_imputation_env.sh
```

## After setup

Activate the environment:

```bash
# Linux / macOS / WSL
source imputation_env/bin/activate
```

Either:

- Open the project in **VS Code** (run `code .` from inside WSL) and select the
  kernel **"Imputation (6G Datasets)"** in any notebook. The Python and Jupyter
  extensions handle activation for you — no manual `source` needed inside
  notebooks. *Recommended.*
- Or run `jupyter lab` and pick the same kernel.

## Reproducing the results

Run the notebooks in order:

1. `notebooks/tuning.ipynb` — hyperparameter search (writes `results/tuned_params.json`)
2. `notebooks/imputation.ipynb` — imputation + reconstruction/fidelity metrics
   (writes `results/csv/*_imputation_results.csv`)
3. `notebooks/downstream.ipynb` — beam-prediction evaluation
   (writes `results/csv/*_downstream_*.csv`)

Cached imputations under `results/` let the downstream step run without re-imputing.