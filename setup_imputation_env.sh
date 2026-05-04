#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Setup script: virtual environment for missing-data imputation experiments.
#
# Stack:
#   - Mean / kNN          -> scikit-learn (SimpleImputer, KNNImputer)
#                            also fancyimpute.SimpleFill / fancyimpute.KNN
#   - MICE                -> scikit-learn IterativeImputer + statsmodels (MICE)
#                            (hyperimpute also bundles a MICE plugin)
#   - SoftImpute          -> fancyimpute.SoftImpute (low-rank, important for CSI)
#   - HyperImpute         -> hyperimpute (deep tabular AutoML imputation)
#   - GRAPE               -> DiffPuter/baselines/GRAPE   (torch-geometric based)
#   - DiffPuter           -> DiffPuter (ICLR 2025 spotlight, diffusion)
#
# After this script finishes you will have:
#   ./imputation_env/                 -- the virtual environment
#   ./DiffPuter/                      -- the cloned repository (pinned commit)
#   ./requirements.txt                -- pinned-version snapshot (pip freeze)
#   ./requirements-base.txt           -- the curated package list this script installed
#   ./requirements-pyg.txt            -- the torch-geometric extras (separate index)
#   A Jupyter kernel called "Imputation (6G Datasets)"
#
# Usage:
#   chmod +x setup_imputation_env.sh
#   ./setup_imputation_env.sh
#
# Tested with Python 3.10 / 3.11 / 3.12 on Linux + macOS + WSL.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------- config ---------------------------------------------------------
ENV_DIR="${ENV_DIR:-imputation_env}"
REPO_DIR="${REPO_DIR:-DiffPuter}"
REPO_URL="https://github.com/hengruizhang98/DiffPuter.git"

# Pinned DiffPuter commit. Bump this manually when you want a newer upstream
# revision. To find your current SHA: cd DiffPuter && git rev-parse HEAD
DIFFPUTER_COMMIT="${DIFFPUTER_COMMIT:-2fa55373655b9e910146d94820fc1012da0dfd75}"

KERNEL_NAME="${KERNEL_NAME:-imputation-6g}"
KERNEL_DISPLAY="Imputation (6G Datasets)"

# Pick a python interpreter. Override with: PYTHON=python3.11 ./setup_imputation_env.sh
PYTHON="${PYTHON:-python3}"

# ---------- helpers --------------------------------------------------------
say()  { printf "\n\033[1;36m==>\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\n\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 || die "Python interpreter '$PYTHON' not found. Install Python 3.10+ or set PYTHON=..."
command -v git      >/dev/null 2>&1 || die "git is required."

PY_VER=$("$PYTHON" -c 'import sys; print("%d.%d"%sys.version_info[:2])')
say "Using $PYTHON (version $PY_VER)"

case "$PY_VER" in
  3.10|3.11|3.12) ;;
  *) warn "Python $PY_VER is outside the tested range (3.10-3.12). Continuing anyway." ;;
esac

# ---------- 1. clone DiffPuter at pinned commit ---------------------------
if [ -d "$REPO_DIR" ]; then
  say "Directory '$REPO_DIR' already exists, skipping clone."
else
  say "Cloning DiffPuter into ./$REPO_DIR at commit $DIFFPUTER_COMMIT"
  git clone --quiet --depth 1 --branch-point "$REPO_URL" "$REPO_DIR" 2>/dev/null || \
    git clone --quiet "$REPO_URL" "$REPO_DIR"
  
  say "Checking out commit $DIFFPUTER_COMMIT"
  git -C "$REPO_DIR" fetch --quiet origin "$DIFFPUTER_COMMIT" 2>/dev/null || true
  git -C "$REPO_DIR" checkout --quiet "$DIFFPUTER_COMMIT" \
    || die "Could not check out $DIFFPUTER_COMMIT in $REPO_DIR. Is the SHA correct?"
  
  say "Removing .git directory to detach from remote repository"
  rm -rf "$REPO_DIR/.git"
fi

# ---------- 2. write requirements files -----------------------------------
say "Writing requirements-base.txt"
cat > requirements-base.txt <<'REQ'
# --- Core scientific stack ---------------------------------------------------
# numpy<2 is pinned because fancyimpute, torch_scatter and several older
# imputation libs still link against the numpy 1.x C ABI. Relaxing this is
# the most common cause of "module compiled against numpy" runtime errors.
numpy<2.0
scipy
# pandas<3 because fancyimpute and parts of hyperimpute were written against
# the pandas 1.x/2.x API; pandas 3 deprecations became hard errors.
pandas<3
scikit-learn
matplotlib
seaborn
statsmodels
tqdm
PyYAML
openpyxl
xlrd
h5py
networkx

# --- DiffPuter extras (from the official requirements/diffputer.txt) ---------
POT
FrEIA
timm

# --- Jupyter -----------------------------------------------------------------
jupyterlab
ipykernel
ipywidgets

# --- PyTorch (CPU build by default; rerun pip with the cuXXX index for GPU) -
torch
torchvision
torchaudio

# --- Baseline imputers -------------------------------------------------------
# fancyimpute -> SoftImpute, KNN, IterativeSVD, SimpleFill
fancyimpute
# hyperimpute pins category_encoders, see DiffPuter's requirements/diffputer.txt
category_encoders==2.5.1.post0
hyperimpute

# torch-geometric is pure-python and lives on PyPI. torch-scatter does NOT
# always have a wheel for every torch+CUDA combo on PyPI, so it is installed
# separately from requirements-pyg.txt against the official PyG wheel index.
torch_geometric
REQ

say "Writing requirements-pyg.txt (torch_scatter, installed against the PyG wheel index)"
cat > requirements-pyg.txt <<'REQ'
# Installed with: pip install -r requirements-pyg.txt -f https://data.pyg.org/whl/torch-<VER>+<TAG>.html
# The setup script fills in <VER> and <TAG> automatically based on the
# installed torch version.
torch_scatter
REQ

# ---------- 3. create venv -------------------------------------------------
if [ -d "$ENV_DIR" ]; then
  say "Virtual env '$ENV_DIR' already exists, reusing it."
else
  say "Creating virtual environment in ./$ENV_DIR"
  "$PYTHON" -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
say "Activated venv: $(which python)"

python -m pip install --upgrade pip setuptools wheel

# ---------- 4. install everything except torch_scatter --------------------
say "Installing from requirements-base.txt"
python -m pip install -r requirements-base.txt

# ---------- 5. torch_scatter against the matching PyG wheel index ---------
TORCH_VER=$(python -c 'import torch; print(torch.__version__)')
TORCH_CUDA=$(python -c 'import torch; print(torch.version.cuda or "cpu")')
say "Detected torch $TORCH_VER (CUDA: $TORCH_CUDA)"

TORCH_TAG="${TORCH_VER%%+*}"   # 2.4.1+cu121 -> 2.4.1
if [ "$TORCH_CUDA" = "cpu" ]; then
  PYG_TAG="cpu"
else
  PYG_TAG="cu$(echo "$TORCH_CUDA" | tr -d .)"   # 12.1 -> cu121
fi
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_TAG}+${PYG_TAG}.html"
say "Installing torch_scatter from $PYG_WHEEL_URL"

if ! python -m pip install -r requirements-pyg.txt -f "$PYG_WHEEL_URL"; then
  warn "No prebuilt torch_scatter wheel for torch=$TORCH_VER ($PYG_TAG)."
  warn "Falling back to a source build (needs a C++ compiler, may take a few minutes)."
  python -m pip install --no-build-isolation torch_scatter || \
    warn "torch_scatter install failed. GRAPE will not run, but the rest of the env is fine."
fi

# ---------- 6. register Jupyter kernel ------------------------------------
say "Registering Jupyter kernel '$KERNEL_DISPLAY'"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

# ---------- 7. quick sanity check -----------------------------------------
say "Running import sanity check"
python - <<'PY'
import importlib, sys
mods = [
    "numpy", "scipy", "pandas", "sklearn", "matplotlib",
    "torch", "torch_geometric",
    "fancyimpute", "hyperimpute",
    "tqdm", "yaml", "h5py", "networkx",
    "ot",   # POT imports as ot
]
optional = ["torch_scatter"]

ok, bad = [], []
for m in mods:
    try:
        importlib.import_module(m); ok.append(m)
    except Exception as e:
        bad.append((m, e))
for m in optional:
    try:
        importlib.import_module(m); ok.append(m + " (optional)")
    except Exception as e:
        print(f"  [optional missing] {m}: {e}")

print("OK :", ", ".join(ok))
if bad:
    print("FAIL:")
    for m, e in bad:
        print(f"  - {m}: {e}")
    sys.exit(1)
PY

# ---------- 8. freeze a fully-pinned snapshot -----------------------------
say "Freezing exact versions to requirements.txt (reproducible snapshot)"
python -m pip freeze > requirements.txt

# ---------- 9. final hint --------------------------------------------------
cat <<EOF

----------------------------------------------------------------------
All done.

Files created:
  requirements-base.txt   -- curated, human-readable list (input)
  requirements-pyg.txt    -- torch_scatter, separate PyG wheel index
  requirements.txt        -- exact pinned versions (pip freeze snapshot)

DiffPuter pinned at: $DIFFPUTER_COMMIT

Activate the env:
    source $ENV_DIR/bin/activate

Launch Jupyter:
    jupyter lab
Then pick the kernel:  "$KERNEL_DISPLAY"

To recreate this env elsewhere with identical versions:
    python -m venv imputation_env
    source imputation_env/bin/activate
    pip install -r requirements.txt -f $PYG_WHEEL_URL

In a notebook, to import GRAPE / DiffPuter from the cloned repo, add:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path('$REPO_DIR').resolve()))
    sys.path.insert(0, str(pathlib.Path('$REPO_DIR/baselines').resolve()))
    sys.path.insert(0, str(pathlib.Path('$REPO_DIR/baselines/GRAPE').resolve()))
----------------------------------------------------------------------
EOF