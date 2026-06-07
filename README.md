# Research project for CSE3000

Imputation experiments on the DeepSense 6G dataset: benchmarking tabular
imputation methods across several missingness mechanisms and rates, evaluated on
reconstruction error, distributional fidelity, and downstream beam-prediction
accuracy.

This repository includes a bootstrap script that creates a Python virtual
environment with all the imputation libraries used in the project. Tested on
Linux, macOS, and Windows via **WSL** (recommended).

## Cloning this repository

This repo uses a git submodule for DiffPuter. Clone with:

```bash
git clone --recurse-submodules https://github.com/KenChan000/ImputationFor6GDatasets
```

If you already cloned without `--recurse-submodules`, initialise the submodule afterwards:

```bash
git submodule update --init
```

## Repository layout

```
ImputationFor6GDatasets/
├── src/
│   ├── common/            # dataset specs, experiment config, tuning, env checks
│   └── imputers/          # imputer wrappers for the full method shortlist
├── external/
│   └── DiffPuter/         # git submodule (GRAPE lives under baselines/GRAPE/)
├── data/                  # DeepSense 6G data — gitignored, you provide it
│   ├── Scenario5/
│   └── Scenario33/
├── results/               # generated figures, CSVs, and tuned_params.json
├── imputation.ipynb       # reconstruction + distributional fidelity 
├── tuning.ipynb           # per-method hyperparameter search
├── downstream.ipynb       # beam-prediction downstream evaluation 
└── setup_imputation_env.sh
```

## Datasets

The DeepSense 6G datasets are **not included** in this repository. They are released for non-commercial academic use by their original authors and must be downloaded directly from the official source.

1. Go to [https://deepsense6g.net](https://deepsense6g.net) and download **Scenario 5** and **Scenario 33**.
2. Place the downloaded files in the `data/` directory at the project root, following this structure:

```
data/
├── Scenario5/
│   └── ...
└── Scenario33/
    └── ...
```

3. If you use these datasets in your own work, cite them as mandated by the DeepSense 6G authors (see their website for the required citation).

## Notebooks

Each notebook has a single path-setup cell at the top where you set
`DATASET = "scenario5"` or `DATASET = "scenario33"` — that is the only cell you
change to switch datasets.

- **`tuning.ipynb`** — per-method hyperparameter search (grid search for the
  cheaper methods, an Optuna joint search for DiffPuter), writing the selected
  configurations to `results/tuned_params.json`.
- **`imputation.ipynb`** — masks the target columns under each mechanism/rate,
  runs every imputer with the tuned parameters, and scores reconstruction
  (RMSE/MAE) and distributional fidelity.
- **`downstream.ipynb`** — feeds the imputed training data into the
  scenario-specific beam predictor and reports top-k accuracy against the clean
  ceiling.

## Imputation methods covered

The environment supports the full method shortlist:

- **Mean** — `scikit-learn` (`SimpleImputer`)
- **kNN** — `scikit-learn` (`KNNImputer`)
- **MICE** — `scikit-learn` (`IterativeImputer`)
- **SoftImpute** — `fancyimpute.SoftImpute` (low-rank, important for CSI data)
- **HyperImpute** — `hyperimpute` package
- **GRAPE** — bundled inside DiffPuter at `external/DiffPuter/baselines/GRAPE/`
- **DiffPuter** — git submodule at `external/DiffPuter/`, imported as source

DiffPuter is a submodule (not pip-installed) because it's a research codebase
rather than a packaged library. Because it's tracked as a submodule, git pins
the **exact commit** for you: `git submodule update --init` checks out the
recorded revision, so everyone builds against the same DiffPuter source.

## Prerequisites

- Python 3.10–3.12 (with `venv` and `pip`)
- git
- A C/C++ compiler (`build-essential` on Debian/Ubuntu) — only needed if a
  prebuilt `torch-scatter` wheel isn't available for your torch+CUDA combo

## Quick start

In a WSL/Linux/macOS terminal, from the project root:

```bash
chmod +x setup_imputation_env.sh
./setup_imputation_env.sh
```

## After setup

Activate the environment:

```bash
# Linux / macOS / WSL
source imputation_env/bin/activate
```

Either:

- Open the project in **VS Code** (run `code .` from inside WSL) and select the kernel **"Imputation (6G Datasets)"** in any notebook. The Python and Jupyter extensions handle activation for you — no manual `source` needed inside notebooks. *Recommended.*
- Or run `jupyter lab` and pick the same kernel.
