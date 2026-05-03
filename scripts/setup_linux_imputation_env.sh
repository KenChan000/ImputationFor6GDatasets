#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/setup_linux_imputation_env.sh
#   bash scripts/setup_linux_imputation_env.sh --skip-grape
#   bash scripts/setup_linux_imputation_env.sh --skip-diffputer
#   bash scripts/setup_linux_imputation_env.sh --skip-all

# By default, both GRAPE and DiffPuter are installed for reproducibility
INSTALL_GRAPE=true
INSTALL_DIFFPUTER=true

for arg in "$@"; do
  case "$arg" in
    --skip-grape)
      INSTALL_GRAPE=false
      ;;
    --skip-diffputer)
      INSTALL_DIFFPUTER=false
      ;;
    --skip-all)
      INSTALL_GRAPE=false
      INSTALL_DIFFPUTER=false
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Supported options: --skip-grape --skip-diffputer --skip-all"
      exit 1
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REPOS_DIR="$PROJECT_DIR/external"
KERNEL_NAME="imputation-6g"
KERNEL_DISPLAY="Python (Imputation6G)"

echo "[1/8] Installing system packages..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git build-essential
else
  echo "apt-get not found. Install python3, python3-venv, python3-pip, git, and build-essential manually."
  exit 1
fi

echo "[2/8] Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/8] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[4/8] Installing notebook and imputation dependencies..."
python -m pip install -r "$PROJECT_DIR/requirements-imputation.txt"

echo "[5/8] Registering Jupyter kernel: $KERNEL_DISPLAY"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

mkdir -p "$REPOS_DIR"

if [ "$INSTALL_GRAPE" = true ]; then
  echo "[6/8] Installing GRAPE from source..."
  if [ ! -d "$REPOS_DIR/GRAPE" ]; then
    git clone https://github.com/maxiaoba/GRAPE "$REPOS_DIR/GRAPE"
  fi
  # GRAPE dependencies from environment.yml
  python -m pip install fancyimpute pyqt5 -q
else
  echo "[6/8] Skipping GRAPE (pass --skip-grape to disable)."
fi

if [ "$INSTALL_DIFFPUTER" = true ]; then
  echo "[7/8] Installing DiffPuter from source..."
  if [ ! -d "$REPOS_DIR/DiffPuter" ]; then
    git clone https://github.com/hengruizhang98/DiffPuter "$REPOS_DIR/DiffPuter"
  fi
  # DiffPuter dependencies
  if [ -f "$REPOS_DIR/DiffPuter/requirements/diffputer.txt" ]; then
    python -m pip install -r "$REPOS_DIR/DiffPuter/requirements/diffputer.txt" -q
  fi
else
  echo "[7/8] Skipping DiffPuter (pass --skip-diffputer to disable)."
fi

echo "[8/8] Done."
echo ""
echo "Next steps:"
echo "  1) Activate env: source $VENV_DIR/bin/activate"
echo "  2) Launch Jupyter: jupyter lab"
echo "  3) In the notebook, select kernel: $KERNEL_DISPLAY"
