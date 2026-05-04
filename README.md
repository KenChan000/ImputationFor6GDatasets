# Research project for CSE3000

Setup for notebook imputation experiments on a tabular 6G dataset.

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

Quick reference:

- **Ubuntu/Debian/WSL:** `sudo apt-get install -y python3 python3-venv python3-pip git build-essential`
- **macOS** (with [Homebrew](https://brew.sh)): `brew install python git` and `xcode-select --install`
- **Windows:** [install WSL](https://learn.microsoft.com/windows/wsl/install) and run the script inside the WSL shell. Native Windows (PowerShell / CMD) is not supported.

## Quick start

In a WSL/Linux/macOS terminal, from the project root:

```bash
chmod +x setup_imputation_env.sh
./setup_imputation_env.sh
```

The script will:

1. Clone the DiffPuter repository to `./DiffPuter` and check out the pinned commit (or reset an existing checkout to it).
2. Write three requirements files: `requirements-base.txt` (curated, with comments), `requirements-pyg.txt` (just `torch_scatter`, separate because it needs the PyG wheel index), and after install a `requirements.txt` snapshot via `pip freeze`.
3. Create a venv at `./imputation_env`.
4. Install the full stack into it (torch defaults to **CPU build**; see *GPU* below).
5. Detect your installed torch + CUDA version and install `torch_scatter` from the matching PyG wheel index. If no prebuilt wheel exists for your combo it falls back to a source build.
6. Register a Jupyter kernel called **"Imputation (6G Datasets)"**.
7. Run an import sanity check.

### Configuration

Override defaults via environment variables:

```bash
PYTHON=python3.11 ./setup_imputation_env.sh                     # specific interpreter
ENV_DIR=my_venv ./setup_imputation_env.sh                       # different venv folder
REPO_DIR=external/DiffPuter ./setup_imputation_env.sh           # different clone location
KERNEL_DISPLAY="My Kernel Name" ./setup_imputation_env.sh       # different kernel display name
DIFFPUTER_COMMIT=abc123... ./setup_imputation_env.sh            # try a different DiffPuter commit
```

### GPU (CUDA) setup

The script installs the **CPU build** of PyTorch by default. If you have an
NVIDIA GPU and want CUDA, after the script finishes:

```bash
source imputation_env/bin/activate

# Pick the index URL that matches your CUDA version (cu121, cu124, cu130, etc.)
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Then reinstall torch_scatter against the matching PyG index
pip install torch_scatter -f https://data.pyg.org/whl/torch-<TORCH_VER>+cu121.html
```

The script prints the exact `-f` URL it used at the end — use the same pattern
with `cu121` swapped in.

### Apple Silicon

PyTorch reports MPS availability separately — DiffPuter and the sklearn-based
methods will use it where supported. `torch_scatter` may not build cleanly on
macOS, in which case GRAPE won't run but the rest of the environment
(Mean / kNN / MICE / SoftImpute / HyperImpute / DiffPuter) works fine.


## Files the script generates

After a successful run, you'll have:

```
imputation_env/             # the virtual environment
DiffPuter/                  # cloned research repo, at the pinned commit
requirements-base.txt       # curated, human-readable list (input)
requirements-pyg.txt        # torch_scatter, separate PyG wheel index
requirements.txt            # exact pinned versions, full transitive snapshot
```

To recreate the environment elsewhere with identical versions:

```bash
python3 -m venv imputation_env
source imputation_env/bin/activate
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-<VER>+<TAG>.html
```

The `-f` URL is printed at the end of the script's run; copy it from there.

## After setup

Activate the environment:

```bash
# Linux / macOS / WSL
source imputation_env/bin/activate
```

Either:

- Open the project in **VS Code** (run `code .` from inside WSL) and select the kernel **"Imputation (6G Datasets)"** in any notebook. The Python and Jupyter extensions handle activation for you — no manual `source` needed inside notebooks. *Recommended.*
- Or run `jupyter lab` and pick the same kernel.

In a notebook, to import GRAPE / DiffPuter from the cloned repo:

```python
import sys, pathlib
root = pathlib.Path("DiffPuter").resolve()
for p in [root, root / "baselines", root / "baselines" / "GRAPE"]:
    sys.path.insert(0, str(p))

# DiffPuter
from dataset import load_dataset, get_eval, mean_std
from diffusion_utils import sample_step, impute_mask, EDMLoss
from model import MLPDiffusion, Model

# GRAPE (only when needed)
from training.gnn_mdi import train_gnn_mdi, out_of_sample_test_gnn_mdi
```
