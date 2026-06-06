# Research project for CSE3000

Setup for notebook imputation experiments on 6G dataset.

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

## Datasets

The DeepSense 6G datasets are **not included** in this repository. They are released for non-commercial academic use by their original authors and must be downloaded directly from the official source.

1. Go to [https://deepsense6g.net](https://deepsense6g.net) and download **Scenario 5** and **Scenario 33**.
2. Place the downloaded files in the `data/` directory at the project root, following this structure:

```
data/
├── scenario5/
│   └── ...
└── scenario33/
    └── ...
```

3. If you use these datasets in your own work, cite them as mandated by the DeepSense 6G authors (see their website for the required citation).

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