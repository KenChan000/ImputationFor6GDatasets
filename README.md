# Research project for CSE3000

Setup for notebook imputation experiments on a tabular 6G dataset.

This repository includes a bootstrap script that creates a Python virtual environment with all the imputation libraries used in the project. Works on **Linux, macOS, and Windows (via WSL or Git Bash)**.

## Imputation methods covered

The environment supports the full method shortlist:

- **Mean / median** — `scikit-learn` (`SimpleImputer`)
- **kNN** — `scikit-learn` (`KNNImputer`)
- **MICE** — `scikit-learn` (`IterativeImputer`)
- **SoftImpute** — `fancyimpute`
- **HyperImpute** — `hyperimpute` package
- **GRAPE** — bundled inside DiffPuter at `external/DiffPuter/baselines/GRAPE/`
- **DiffPuter** — cloned to `external/DiffPuter` and imported directly

DiffPuter is cloned (not pip-installed) because it's a research codebase rather than a packaged library. The clone lives under `external/` and is pinned to a specific commit for reproducibility (see *Pinned commits* below).

## Prerequisites

The setup script does **not** install system packages — it checks that they exist and tells you what's missing. Before running it, install:

- Python ≥ 3.9 (with `venv` and `pip`)
- git
- A C/C++ compiler (some pip wheels need it)

The script prints exact install commands for your OS if anything is missing. Quick reference:

- **Ubuntu/Debian:** `sudo apt-get install -y python3 python3-venv python3-pip git build-essential`
- **Fedora/RHEL:** `sudo dnf install -y python3 python3-pip git gcc gcc-c++ make`
- **Arch:** `sudo pacman -S python python-pip git base-devel`
- **macOS** (with [Homebrew](https://brew.sh)): `brew install python git` and `xcode-select --install`
- **Windows:** [install WSL](https://learn.microsoft.com/windows/wsl/install) and run the script inside the WSL shell. Git Bash also works but is not recommended for this stack.

## Quick start

1. Open a terminal in this folder.
2. (NVIDIA GPU users) check your CUDA version with `nvidia-smi` and set `TORCH_CUDA` accordingly:

   ```bash
   export TORCH_CUDA=cu121   # or cu118, cu124 — match your driver
   ```

   The default is `cu121`. For CPU-only installs, see *CPU-only PyTorch* below.

3. Run:

   ```bash
   bash scripts/setup_imputation_env.sh
   ```

On the first run the script clones DiffPuter and pins the current HEAD of its default branch into `external/repo-locks.txt`. Subsequent runs check out exactly that commit, so the environment stays reproducible across machines and over time.

### Optional flags

```bash
# Skip cloning DiffPuter (Python deps only)
bash scripts/setup_imputation_env.sh --skip-diffputer

# Skip torch-geometric and the GRAPE baseline (saves ~1–2 GB if you don't need GRAPE)
bash scripts/setup_imputation_env.sh --skip-grape

# Force a specific Python interpreter
PYTHON_BIN=python3.11 bash scripts/setup_imputation_env.sh

# Refresh the pinned commit to current HEAD on the default branch
bash scripts/setup_imputation_env.sh --update-locks
```

### CPU-only PyTorch

If you don't have an NVIDIA GPU, override the CUDA index with the CPU one:

```bash
TORCH_CUDA=cpu bash scripts/setup_imputation_env.sh --skip-grape
```

GRAPE relies on torch-geometric companion ops (`torch-scatter`, `torch-sparse`) that are awkward to install in CPU-only mode, so `--skip-grape` is recommended on CPU machines.

## Pinned commits

The pinned SHA in `external/repo-locks.txt` is intentionally sticky — it won't change unless you ask. To bump it (e.g. to pull in upstream fixes), run with `--update-locks` and re-run any experiments whose results you care about, since baseline behavior may have changed.

## After setup

1. Activate the environment:

   ```bash
   # Linux / macOS / WSL
   source .venv/bin/activate

   # Git Bash on Windows
   source .venv/Scripts/activate
   ```

2. Either:

   - Open the project in **VS Code** and select kernel `Python (Imputation6G)` in any notebook, or
   - Run `jupyter lab` and pick the same kernel.

## Notes

- Native Windows (PowerShell / CMD) is not supported. Use WSL.
- On Apple Silicon, the sanity check reports MPS availability in addition to CUDA. Set `TORCH_CUDA=cpu` since CUDA isn't available on macOS.
- The setup script's final sanity-check block prints which methods are correctly importable. Use it to confirm everything installed.