#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Setup script: virtual environment for missing-data imputation experiments.
#
# Stack:
#   - Mean / kNN          -> scikit-learn (SimpleImputer, KNNImputer)
#   - MICE                -> scikit-learn IterativeImputer + statsmodels
#   - SoftImpute          -> fancyimpute.SoftImpute (low-rank)
#   - HyperImpute         -> hyperimpute (deep tabular AutoML imputation)
#   - GRAPE               -> external/DiffPuter/baselines/GRAPE (torch-geometric)
#   - DiffPuter           -> external/DiffPuter (ICLR 2025 spotlight, diffusion)
#
# This script lives in scripts/ and anchors all paths to the repo root, so it
# works regardless of the directory you launch it from. It creates:
#   <root>/imputation_env/            -- the virtual environment
#   <root>/external/DiffPuter/        -- the cloned repo (pinned commit)
#   <root>/requirements/base.txt      -- curated package list this script installs
#   <root>/requirements/pyg.txt       -- torch-geometric extras (separate index)
#   <root>/requirements/lock.txt      -- pinned-version snapshot (pip freeze)
#   A Jupyter kernel called "Imputation (6G Datasets)"
#
# Usage (from anywhere):
#   chmod +x scripts/setup_imputation_env.sh
#   ./scripts/setup_imputation_env.sh
#
# Tested with Python 3.10 / 3.11 / 3.12 on Linux + macOS + WSL.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------- locate repo root and cd there ----------------------------------
# Resolve the directory this script lives in (scripts/), then its parent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------- config ---------------------------------------------------------
ENV_DIR="${ENV_DIR:-imputation_env}"
REPO_DIR="${REPO_DIR:-external/DiffPuter}"          # vendored under external/
REPO_URL="https://github.com/hengruizhang98/DiffPuter.git"
REQ_DIR="${REQ_DIR:-requirements}"                  # dedicated requirements/ dir

# Pinned DiffPuter commit. Bump manually for a newer upstream revision.
# Find current SHA: git -C external/DiffPuter rev-parse HEAD
DIFFPUTER_COMMIT="${DIFFPUTER_COMMIT:-2fa55373655b9e910146d94820fc1012da0dfd75}"

KERNEL_NAME="${KERNEL_NAME:-imputation-6g}"
KERNEL_DISPLAY="Imputation (6G Datasets)"

PYTHON="${PYTHON:-python3}"

# ---------- helpers --------------------------------------------------------
say()  { printf "\n\033[1;36m==>\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\n\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 || die "Python interpreter '$PYTHON' not found. Install Python 3.10+ or set PYTHON=..."
command -v git      >/dev/null 2>&1 || die "git is required."

say "Repo root: $REPO_ROOT"

PY_VER=$("$PYTHON" -c 'import sys; print("%d.%d"%sys.version_info[:2])')
say "Using $PYTHON (version $PY_VER)"

case "$PY_VER" in
  3.10|3.11|3.12) ;;
  *) warn "Python $PY_VER is outside the tested range (3.10-3.12). Continuing anyway." ;;
esac

mkdir -p "$REQ_DIR"
mkdir -p "$(dirname "$REPO_DIR")"   # ensure external/ exists before clone

REQ_BASE="$REQ_DIR/base.txt"
REQ_PYG="$REQ_DIR/pyg.txt"
REQ_FREEZE="$REQ_DIR/lock.txt"

# ---------- 1. clone DiffPuter at pinned commit ---------------------------
if [ -d "$REPO_DIR" ]; then
  say "Directory '$REPO_DIR' already exists, skipping clone."
else
  say "Cloning DiffPuter into ./$REPO_DIR"
  git clone --quiet "$REPO_URL" "$REPO_DIR"

  say "Checking out commit $DIFFPUTER_COMMIT"
  git -C "$REPO_DIR" fetch --quiet origin "$DIFFPUTER_COMMIT" 2>/dev/null || true
  git -C "$REPO_DIR" checkout --quiet "$DIFFPUTER_COMMIT" \
    || die "Could not check out $DIFFPUTER_COMMIT in $REPO_DIR. Is the SHA correct?"

  say "Removing .git directory to detach from remote repository"
  rm -rf "$REPO_DIR/.git"
fi

# ---------- 2. write requirements files -----------------------------------
say "Writing $REQ_BASE"
cat > "$REQ_BASE" <<'REQ'
# --- Core scientific stack ---------------------------------------------------
# numpy<2 is pinned because fancyimpute, torch_scatter and several older
# imputation libs still link against the numpy 1.x C ABI. Relaxing this is
# the most common cause of "module compiled against numpy" runtime errors.
numpy<2.0
scipy
# pandas<3 because fancyimpute and parts of hyperimpute were written against
# the pandas 1.x/2.x API; pandas 3 deprecations became hard errors.
pandas<3
# scikit-learn pinned <1.7 because hyperimpute 0.1.17 uses LogisticRegression's
# `multi_class` parameter, which was deprecated in sklearn 1.5 and removed in
# 1.7. This is the most common breakage when reinstalling the env. Loosen this
# only if you've upgraded hyperimpute to a version that supports newer sklearn.
scikit-learn>=1.4,<1.7
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
# Upper bound on torch because PyG wheels (torch_scatter) often lag behind the
# latest torch release. If torch_scatter install fails, check PyG's compat
# table at https://data.pyg.org/whl/ and adjust this range.
torch>=2.0,<2.5
torchvision
torchaudio

# --- Baseline imputers -------------------------------------------------------
# fancyimpute -> SoftImpute, KNN, IterativeSVD, SimpleFill
fancyimpute>=0.7,<0.8
# hyperimpute pinned to the exact version known to work with the rest of this
# stack. Bumping this requires re-checking the sklearn pin above.
hyperimpute==0.1.17
# category_encoders pinned because hyperimpute requires this exact version,
# see DiffPuter's requirements/diffputer.txt.
category_encoders==2.5.1.post0

# torch-geometric is pure-python and lives on PyPI. torch-scatter does NOT
# always have a wheel for every torch+CUDA combo on PyPI, so it is installed
# separately from pyg.txt against the official PyG wheel index.
torch_geometric
REQ

say "Writing $REQ_PYG (torch_scatter, installed against the PyG wheel index)"
cat > "$REQ_PYG" <<'REQ'
# Installed with: pip install -r requirements/pyg.txt -f https://data.pyg.org/whl/torch-<VER>+<TAG>.html
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
say "Installing from $REQ_BASE"
python -m pip install -r "$REQ_BASE"

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

if ! python -m pip install -r "$REQ_PYG" -f "$PYG_WHEEL_URL"; then
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

import sklearn
from packaging.version import Version
sk_ver = Version(sklearn.__version__)
if sk_ver >= Version("1.7"):
    print(f"WARN: sklearn {sklearn.__version__} >= 1.7, hyperimpute 0.1.17 may break "
      f"on LogisticRegression(multi_class=...). Check requirements/base.txt pins.")
elif sk_ver < Version("1.4"):
    print(f"WARN: sklearn {sklearn.__version__} < 1.4, behaviour may differ from tested range.")
else:
    print(f"sklearn {sklearn.__version__} is within the supported range (>=1.4,<1.7).")
PY

# ---------- 8. freeze a fully-pinned snapshot -----------------------------
say "Freezing exact versions to $REQ_FREEZE (reproducible snapshot)"
python -m pip freeze > "$REQ_FREEZE"

# ---------- 9. final hint --------------------------------------------------
cat <<EOF

----------------------------------------------------------------------
All done.

Files created (under $REPO_ROOT):
  requirements/base.txt   -- curated, human-readable list (input)
  requirements/pyg.txt    -- torch_scatter, separate PyG wheel index
  requirements/lock.txt   -- exact pinned versions (pip freeze snapshot)

DiffPuter pinned at: $DIFFPUTER_COMMIT  (in external/DiffPuter)

Activate the env:
    source $ENV_DIR/bin/activate

Launch Jupyter:
    jupyter lab
Then pick the kernel:  "$KERNEL_DISPLAY"

To recreate this env elsewhere with identical versions:
    python -m venv imputation_env
    source imputation_env/bin/activate
    pip install -r requirements/lock.txt -f $PYG_WHEEL_URL

The notebook's setup cell already puts external/DiffPuter (and its
baselines/GRAPE) on sys.path, so no manual path injection is needed.
----------------------------------------------------------------------
EOF