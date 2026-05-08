# Research project for CSE3000

Setup for notebook imputation experiments on 6G dataset.

This repository includes a bootstrap script that creates a Python virtual
environment with all the imputation libraries used in the project. Tested on
Linux, macOS, and Windows via **WSL** (recommended).

## Imputation methods covered

The environment supports the full method shortlist:

- **Mean** — `scikit-learn` (`SimpleImputer`)
- **kNN** — `scikit-learn` (`KNNImputer`)
- **MICE** — `scikit-learn` (`IterativeImputer`)
- **SoftImpute** — `fancyimpute.SoftImpute` (low-rank, important for CSI data)
- **HyperImpute** — `hyperimpute` package
- **GRAPE** — bundled inside DiffPuter at `DiffPuter/baselines/GRAPE/`
- **DiffPuter** — cloned to `DiffPuter/` at the project root and imported as source

DiffPuter is cloned (not pip-installed) because it's a research codebase rather
than a packaged library. The clone is **pinned to a specific commit** for
reproducibility — see *Pinned DiffPuter commit* below.

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

