#!/usr/bin/env bash
set -euo pipefail

# Setup for the 6G imputation research environment (CSE3000).
# Works on: Linux, macOS, Windows (via WSL or Git Bash). Native Windows: use WSL.
#
# Methods supported:
#   - Mean / median / kNN / MICE   (scikit-learn)
#   - SoftImpute                   (fancyimpute)
#   - HyperImpute                  (hyperimpute)
#   - GRAPE                        (via DiffPuter; needs torch-scatter / torch-sparse)
#   - DiffPuter                    (cloned repo)
#
# Usage:
#   bash scripts/setup_imputation_env.sh
#   bash scripts/setup_imputation_env.sh --skip-grape
#   bash scripts/setup_imputation_env.sh --skip-diffputer
#   bash scripts/setup_imputation_env.sh --update-locks
#   PYTHON_BIN=python3.11 bash scripts/setup_imputation_env.sh
#   TORCH_CUDA=cu126 bash scripts/setup_imputation_env.sh   # override CUDA version

# ---- Configuration ---------------------------------------------------------

DIFFPUTER_REPO="https://github.com/hengruizhang98/DiffPuter"
DIFFPUTER_BRANCH="main"

# We pin torch to a version that has confirmed PyG wheel availability.
# PyG publishes torch-scatter/torch-sparse wheels at https://data.pyg.org/whl/.
# torch 2.7.1 has wheels for cu118, cu126, cu128 — broad coverage.
TORCH_VERSION="2.7.1"

# CUDA target. cu128 is the modern default. Acceptable for torch 2.7.1:
#   cpu, cu118, cu126, cu128
# Run `nvidia-smi` to see your driver's CUDA version. Driver 560+ supports cu128.
TORCH_CUDA="${TORCH_CUDA:-cu128}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- Argument parsing ------------------------------------------------------

INSTALL_DIFFPUTER=true
INSTALL_GRAPE=true
UPDATE_LOCKS=false

for arg in "$@"; do
  case "$arg" in
    --skip-diffputer) INSTALL_DIFFPUTER=false ;;
    --skip-grape)     INSTALL_GRAPE=false ;;
    --update-locks)   UPDATE_LOCKS=true ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REPOS_DIR="$PROJECT_DIR/external"
LOCK_FILE="$REPOS_DIR/repo-locks.txt"
KERNEL_NAME="imputation-6g"
KERNEL_DISPLAY="Python (Imputation6G)"

mkdir -p "$REPOS_DIR"

# ---- OS detection ----------------------------------------------------------

detect_os() {
  case "$(uname -s)" in
    Linux*)
      if grep -qi microsoft /proc/version 2>/dev/null; then echo "wsl"; else echo "linux"; fi ;;
    Darwin*)              echo "macos" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows-bash" ;;
    *)                    echo "unknown" ;;
  esac
}

OS="$(detect_os)"

# ---- Prerequisite checks ---------------------------------------------------

print_install_hint() {
  echo "Missing tools: $*"
  case "$OS" in
    linux|wsl) echo "Ubuntu/Debian: sudo apt-get install -y python3 python3-venv python3-pip git build-essential" ;;
    macos)     echo "macOS: brew install python git && xcode-select --install" ;;
    windows-bash) echo "Use WSL: https://learn.microsoft.com/windows/wsl/install" ;;
    *)         echo "Install python3 (>=3.10), pip, venv, git, and a C/C++ compiler." ;;
  esac
}

check_prereqs() {
  echo "Detected OS: $OS"
  if [ "$OS" = "windows-bash" ]; then
    echo "WARNING: Native Windows detected. Strongly recommend WSL."
    read -r -p "Continue anyway? [y/N] " reply
    case "$reply" in y|Y) ;; *) exit 0 ;; esac
  fi

  local missing=()
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    missing+=("$PYTHON_BIN")
  else
    local pyver major minor
    pyver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    major="${pyver%%.*}"; minor="${pyver##*.}"
    if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
      echo "ERROR: $PYTHON_BIN is $pyver. torch $TORCH_VERSION requires Python >= 3.10."
      exit 1
    fi
  fi

  command -v "$PYTHON_BIN" >/dev/null 2>&1 && \
    "$PYTHON_BIN" -c 'import venv' >/dev/null 2>&1 || missing+=("python3-venv")
  command -v git >/dev/null 2>&1 || missing+=("git")
  command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || \
    command -v clang >/dev/null 2>&1 || missing+=("C/C++ compiler")

  if [ ${#missing[@]} -gt 0 ]; then print_install_hint "${missing[@]}"; exit 1; fi
  echo "All prerequisites found."
}

# ---- Lock file helpers -----------------------------------------------------

read_lock() { [ -f "$LOCK_FILE" ] && awk -v n="$1" '$1==n {print $2}' "$LOCK_FILE"; }
write_lock() {
  local tmp; tmp="$(mktemp)"
  [ -f "$LOCK_FILE" ] && awk -v n="$1" '$1!=n' "$LOCK_FILE" > "$tmp"
  echo "$1 $2" >> "$tmp"; mv "$tmp" "$LOCK_FILE"
}

clone_at_locked_commit() {
  local repo_url="$1" target_dir="$2" branch="$3" name="$4"
  if [ ! -d "$target_dir/.git" ]; then
    git clone "$repo_url" "$target_dir"
  else
    (cd "$target_dir" && git fetch --all --tags --quiet)
  fi

  local locked_sha; locked_sha="$(read_lock "$name")"
  if [ "$UPDATE_LOCKS" = true ] || [ -z "$locked_sha" ]; then
    locked_sha="$(cd "$target_dir" && git rev-parse "origin/$branch")"
    echo "  Pinning $name to $locked_sha"
    write_lock "$name" "$locked_sha"
  else
    echo "  Using locked $name commit: $locked_sha"
  fi
  (cd "$target_dir" && git checkout --quiet "$locked_sha")
}

# ---- Steps -----------------------------------------------------------------

echo "[1/7] Checking prerequisites..."
check_prereqs

echo ""
echo "[2/7] Creating virtual environment at $VENV_DIR ..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
if   [ -f "$VENV_DIR/bin/activate" ];     then source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then source "$VENV_DIR/Scripts/activate"
else echo "ERROR: cannot find venv activate script."; exit 1
fi

echo "[3/7] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[4/7] Installing classical imputation deps..."
python -m pip install -r "$PROJECT_DIR/requirements-imputation.txt"

echo "[5/7] Installing PyTorch ${TORCH_VERSION} (CUDA=${TORCH_CUDA})..."
# Pin to a version with confirmed PyG wheel coverage. Avoids the "latest torch
# but no torch-scatter wheels yet" trap.
if [ "$TORCH_CUDA" = "cpu" ]; then
  python -m pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/cpu"
else
  python -m pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

if [ "$INSTALL_GRAPE" = true ]; then
  echo "  Installing torch-geometric..."
  python -m pip install torch-geometric

  echo "  Installing torch-scatter / torch-sparse from PyG wheel index..."
  PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA}.html"
  echo "  Index: $PYG_INDEX"
  if ! python -m pip install torch-scatter torch-sparse -f "$PYG_INDEX"; then
    echo ""
    echo "  ERROR: Could not install torch-scatter / torch-sparse."
    echo "  This usually means TORCH_CUDA=$TORCH_CUDA isn't published for torch $TORCH_VERSION."
    echo ""
    echo "  Valid CUDA values for torch $TORCH_VERSION: cpu, cu118, cu126, cu128"
    echo "  Run with one of those, e.g.:"
    echo "    TORCH_CUDA=cu126 bash scripts/setup_imputation_env.sh"
    echo ""
    echo "  Or skip GRAPE entirely:"
    echo "    bash scripts/setup_imputation_env.sh --skip-grape"
    exit 1
  fi
else
  echo "  Skipping torch-geometric (--skip-grape)."
fi

echo "[6/7] Registering Jupyter kernel: $KERNEL_DISPLAY"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

echo "[7/7] Cloning DiffPuter..."
if [ "$INSTALL_DIFFPUTER" = true ]; then
  clone_at_locked_commit "$DIFFPUTER_REPO" "$REPOS_DIR/DiffPuter" "$DIFFPUTER_BRANCH" "DiffPuter"

  if [ -f "$REPOS_DIR/DiffPuter/requirements/diffputer.txt" ]; then
    python -m pip install -r "$REPOS_DIR/DiffPuter/requirements/diffputer.txt"
  fi
fi

echo ""
echo "Sanity checks..."
python - <<'PY'
import sys, platform
print(f"Python:   {sys.version.split()[0]}")
print(f"Platform: {platform.system()} ({platform.machine()})")

def check(label, importer):
    try:
        v = importer()
        print(f"  [OK]  {label}" + (f" {v}" if v else ""))
    except Exception as e:
        print(f"  [MISS] {label}  ({type(e).__name__}: {e})")

print("Imputation methods:")
check("scikit-learn (mean/kNN/MICE)", lambda: __import__("sklearn").__version__)
check("fancyimpute (SoftImpute)",     lambda: __import__("fancyimpute").__version__)
check("hyperimpute",                  lambda: getattr(__import__("hyperimpute"), "__version__", ""))

print("Deep learning stack:")
check("torch", lambda: __import__("torch").__version__)
try:
    import torch
    print(f"        CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"        Device: {torch.cuda.get_device_name(0)}")
except ImportError: pass
check("torch-geometric", lambda: __import__("torch_geometric").__version__)
check("torch-scatter (GRAPE)", lambda: "")
check("torch-sparse (GRAPE)",  lambda: "")
PY

echo ""
echo "Done."
[ -f "$LOCK_FILE" ] && { echo "Pinned commits:"; sed 's/^/  /' "$LOCK_FILE"; }
echo ""
echo "Next: source $VENV_DIR/bin/activate, open notebook in VS Code or Jupyter."
echo "Kernel: $KERNEL_DISPLAY"