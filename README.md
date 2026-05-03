Research project for CSE3000

Linux setup for notebook imputation experiments

This repository includes a bootstrap script to set up Python, Jupyter, and core imputation libraries on a clean Linux machine.

What is installed
- Mean and kNN imputation support via scikit-learn
- MICE and SoftImpute workflows via hyperimpute
- JupyterLab + ipykernel for notebook execution
- Optional source installs for GRAPE and DiffPuter

Quick start
1. Open a terminal in this folder.
2. Run:

```bash
bash scripts/setup_linux_imputation_env.sh
```

This will install all imputation methods including GRAPE and DiffPuter research frameworks.
The setup script clones those repositories as source snapshots and removes their nested .git folders, so they will not appear as separate git repositories in VS Code.

Optional: Skip research frameworks that require external repo cloning:

```bash
# Skip GRAPE
bash scripts/setup_linux_imputation_env.sh --skip-grape

# Skip DiffPuter
bash scripts/setup_linux_imputation_env.sh --skip-diffputer

# Skip both GRAPE and DiffPuter
bash scripts/setup_linux_imputation_env.sh --skip-all
```

After setup
1. Activate the environment:

```bash
source .venv/bin/activate
```

2. Start Jupyter:

```bash
jupyter lab
```

3. In the notebook, select kernel: Python (Imputation6G).

Notes
- The script assumes an Ubuntu or Debian-style system with apt-get.
- If you use another distro, install equivalent packages manually: python3, python3-venv, python3-pip, git, build-essential.
