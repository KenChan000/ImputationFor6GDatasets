# Benchmarking Tabular Imputation Methods for 6G Wireless Datasets

> **How do different tabular imputation techniques compare when addressing missing values in 6G datasets?**

A systematic benchmark of seven imputation methods — Mean, kNN, MICE, SoftImpute,
HyperImpute, GRAPE, and DiffPuter — across four missingness mechanisms (MCAR, MAR,
MNAR, and structured row-wise MCAR) at three rates (10 %, 30 %, 50 %), evaluated on
reconstruction error, distributional fidelity, and downstream beam-prediction accuracy
on two DeepSense 6G scenarios.

This repository includes a bootstrap script that creates a Python virtual
environment with all the imputation libraries used in the project. Tested on
Linux, macOS, and Windows via **WSL** (recommended).

## Research questions

- **RQ1** — How do imputation methods compare in reconstruction error (RMSE/MAE)
  across four missingness mechanisms (MCAR, MAR, MNAR, row-wise MCAR) and three
  missingness rates (10 %, 30 %, 50 %) in 6G datasets?
- **RQ2** — How well do imputation methods preserve the statistical properties
  (mean, variance, distribution, inter-feature correlations) of the original 6G data?
- **RQ3** — To what extent does the choice of imputation method affect the performance
  of a downstream machine-learning task (beam prediction) on 6G data?

## Paper

This repository accompanies the Bachelor's thesis:

> Kenneth Chan, *"Data quality improvement through data cleaning and augmentation
> methods"*, CSE3000 Research Project, EEMCS, Delft University of Technology,
> June 2026.
> Supervisors: Rihan Hai, Yuandou Wong. Committee: Julian Urbano.

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

Run the notebooks in order. Each has a single path-setup cell at the top where you set
`DATASET = "scenario5"` or `DATASET = "scenario33"` — that is the only cell you change
to switch datasets.

- **`tuning.ipynb`** — per-method hyperparameter search (grid search for kNN and
  SoftImpute; Optuna joint search for DiffPuter and GRAPE), writing the selected
  configurations to `results/tuned_params.json`.
- **`imputation.ipynb`** — masks the target columns under each mechanism/rate,
  runs every imputer with the tuned parameters, and scores reconstruction
  (RMSE/MAE) and distributional fidelity (RQ1/RQ2).
- **`downstream.ipynb`** — feeds the imputed training data into the
  scenario-specific beam predictor and reports Top-K accuracy against the clean
  ceiling (RQ3).

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
