from __future__ import annotations

import importlib
from importlib import metadata
from typing import Callable


# Module name (for import) -> distribution name (for version lookup).
# These differ for some packages (sklearn vs scikit-learn, yaml vs PyYAML,
# ot vs POT).
REQUIRED_MODULES: dict[str, str] = {
    # core scientific
    "numpy":        "numpy",
    "pandas":       "pandas",
    "scipy":        "scipy",
    "sklearn":      "scikit-learn",
    "matplotlib":   "matplotlib",
    "statsmodels":  "statsmodels",
    # baselines + SOTA
    "fancyimpute":  "fancyimpute",   # Mean (SimpleFill), kNN, SoftImpute
    "hyperimpute":  "hyperimpute",   # HyperImpute + MICE plugin
    # deep learning + GRAPE deps
    "torch":            "torch",
    "torch_geometric":  "torch_geometric",
    "h5py":             "h5py",
    "networkx":         "networkx",
    # DiffPuter deps
    "ot":     "POT",                 # POT distributes as POT, imports as ot
    "FrEIA":  "FrEIA",
    "timm":   "timm",
    "yaml":   "PyYAML",
    # notebook
    "ipykernel": "ipykernel",
}

# Nice-to-have; the env still works without them.
OPTIONAL_MODULES: dict[str, str] = {
    "torch_scatter": "torch_scatter",   # GRAPE only
}

# Smoke-test imports for the imputation APIs actually used downstream.
API_CHECKS: list[tuple[str, Callable[[], object]]] = [
    ("Mean / kNN (sklearn)",
     lambda: __import__("sklearn.impute", fromlist=["SimpleImputer", "KNNImputer"])),
    ("MICE (sklearn IterativeImputer)",
     lambda: (
         __import__("sklearn.experimental", fromlist=["enable_iterative_imputer"]),
         __import__("sklearn.impute",        fromlist=["IterativeImputer"]),
     )),
    ("SoftImpute / KNN / SimpleFill (fancyimpute)",
     lambda: __import__("fancyimpute", fromlist=["SoftImpute", "KNN", "SimpleFill"])),
    ("HyperImpute (hyperimpute.plugins.imputers.Imputers)",
     lambda: __import__("hyperimpute.plugins.imputers", fromlist=["Imputers"])),
    ("PyTorch Geometric (for GRAPE)",
     lambda: __import__("torch_geometric")),
]

# Local research-repo modules (importable only once the setup cell has put
# external/DiffPuter + baselines/GRAPE on sys.path).
REPO_CHECKS: list[tuple[str, str, list[str]]] = [
    ("DiffPuter dataset",   "dataset",          ["load_dataset", "get_eval", "mean_std"]),
    ("DiffPuter diffusion", "diffusion_utils",  ["sample_step", "impute_mask", "EDMLoss"]),
    ("DiffPuter model",     "model",            ["MLPDiffusion", "Model"]),
    ("GRAPE training",      "training.gnn_mdi", ["train_gnn_mdi"]),
]


def _check_module(module_name: str, package_name: str) -> tuple[str, str]:
    """Return ("OK", version) or ("MISSING", error-string)."""
    try:
        importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001 - we want to report any import failure
        return "MISSING", str(e)
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        version = "unknown"
    return "OK", version


def check_environment(verbose: bool = True) -> dict:
    def log(*args):
        if verbose:
            print(*args)

    required: dict[str, dict] = {}
    missing: list[str] = []

    log("Required packages")
    log("-" * 60)
    for module_name, package_name in REQUIRED_MODULES.items():
        status, info = _check_module(module_name, package_name)
        required[package_name] = {"status": status, "info": info}
        marker = "\u2713" if status == "OK" else "\u2717"
        log(f"  {marker} {package_name:18} {status:8} {info}")
        if status == "MISSING":
            missing.append(package_name)

    optional: dict[str, dict] = {}
    log("\nOptional packages")
    log("-" * 60)
    for module_name, package_name in OPTIONAL_MODULES.items():
        status, info = _check_module(module_name, package_name)
        optional[package_name] = {"status": status, "info": info}
        marker = "\u2713" if status == "OK" else "\u25cb"
        log(f"  {marker} {package_name:18} {status:8} {info}")

    apis: dict[str, dict] = {}
    log("\nSmoke-test imports for imputation APIs")
    log("-" * 60)
    for label, fn in API_CHECKS:
        try:
            fn()
            apis[label] = {"status": "OK", "error": None}
            log(f"  \u2713 {label}: OK")
        except Exception as e:  # noqa: BLE001
            apis[label] = {"status": "FAILED", "error": str(e)}
            log(f"  \u2717 {label}: FAILED | {e}")

    repos: dict[str, dict] = {}
    log("\nLocal repo modules (DiffPuter / GRAPE)")
    log("-" * 60)
    for label, mod, names in REPO_CHECKS:
        try:
            m = importlib.import_module(mod)
            for n in names:
                getattr(m, n)
            repos[label] = {"status": "OK", "error": None}
            log(f"  \u2713 {label}: OK ({mod})")
        except Exception as e:  # noqa: BLE001
            repos[label] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
            log(f"  \u2717 {label}: FAILED | {type(e).__name__}: {e}")

    if missing:
        log(f"\n\u26a0  Missing required packages: {', '.join(missing)}")
        log("   Re-run: pip install -r requirements/base.txt")
    else:
        log("\nAll required packages present.")

    return {
        "required": required,
        "optional": optional,
        "apis": apis,
        "repos": repos,
        "missing": missing,
        "ok": not missing,
    }


if __name__ == "__main__":
    import sys
    result = check_environment(verbose=True)
    sys.exit(0 if result["ok"] else 1)