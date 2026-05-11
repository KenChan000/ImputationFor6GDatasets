"""
GRAPE adapter for the imputation benchmark framework.
 
Reference: You et al. (2020), "Handling Missing Data with Graph
Representation Learning", NeurIPS 2020.
Repo: https://github.com/maxiaoba/GRAPE
Fork used: DiffPuter/baselines/GRAPE (Zhang et al., ICLR 2025), which
exposes `get_data_fix_mask` to bypass GRAPE's native random edge split
and use a caller-supplied missingness mask instead.
 
GRAPE operates on a bipartite graph: row-nodes <-> column-nodes, with
observed entries as edges. Imputation = predicting values for the
"test" edges (the positions the caller marked missing).
 
This adapter:
  1. Treats NaN positions in X as the test edges and non-NaN as train
     edges (mask convention: True = observed).
  2. Calls `get_data_fix_mask(df_X, mask)` to build the PyG Data object
     with the fixed mask (not GRAPE's default random 70/30 split).
  3. Calls `train_gnn_mdi(data, args, log_path, device, return_filled_X=True)`,
     which returns the fully-imputed matrix directly. No pickle digging.
  4. Restores originally-observed cells exactly (paranoia against
     floating-point drift, matching the DiffPuter adapter convention).
 
Important: the caller is expected to pass already-standardized input
(mean 0, var 1 per column). GRAPE training is scale-sensitive.
 
Note on slow training: GRAPE defaults to 20000 epochs (per the paper).
On a 2300 x 70 matrix this is ~5-15 minutes per call on GPU. For a full
benchmark sweep (3 scenarios x 3 proportions x 10 seeds = 90 calls)
budget several hours and consider reducing `epochs` for exploratory runs.
"""
 
from __future__ import annotations
 
import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
 
import numpy as np
import pandas as pd

def _ensure_grape_on_path(grape_root: str | None = None) -> Path:
    """Find DiffPuter/baselines/GRAPE on disk and add it to sys.path."""
    repo_root = (
        Path(grape_root)
        if grape_root is not None
        else Path(__file__).resolve().parents[1] / "DiffPuter" / "baselines" / "GRAPE"
    ).resolve()

    if not (repo_root / "training" / "gnn_mdi.py").is_file() or not (repo_root / "uci" / "uci_data.py").is_file():
        raise FileNotFoundError(
            f"GRAPE repo not found at {repo_root}. "
            "Pass grape_root='/path/to/DiffPuter/baselines/GRAPE' explicitly."
        )

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root

def _default_grape_args(seed: int, epochs: int) -> SimpleNamespace:
    """
    Build the args namespace train_gnn_mdi expects. Values match the
    argparse defaults in DiffPuter/baselines/GRAPE/train_mdi.py and
    GRAPE_benchmark.py, with seed and epochs overridden per-call.
    """
    return SimpleNamespace(
        # Model architecture
        model_types="EGSAGE_EGSAGE_EGSAGE",
        post_hiddens=None,
        concat_states=False,
        norm_embs=None,
        aggr="mean",
        node_dim=64,
        edge_dim=64,
        edge_mode=1,
        gnn_activation="relu",
        impute_hiddens="64",
        impute_activation="relu",
        # Optimization
        epochs=epochs,
        opt="adam",
        opt_scheduler="none",
        opt_restart=0,
        opt_decay_step=1000,
        opt_decay_rate=0.9,
        dropout=0.0,
        weight_decay=0.0,
        lr=1e-3,
        # Edge handling
        known=0.7,
        auto_known=False,
        loss_mode=0,
        valid=0.0,
        # Misc fields the training loop reads
        seed=seed,
        save_model=False,
        save_prediction=False,
        transfer_dir=None,
        transfer_extra="",
        mode="train",
        # Fields GRAPE checks via hasattr; set them so it doesn't matter
        split_sample=0.0,
        split_by="y",
        split_train=False,
        split_test=False,
        train_y=0.7,
        node_mode=0,
        train_edge=0.7,
        # Domain / data identifiers (some code paths read them)
        domain="uci",
        data="custom",
        log_dir="0",
        comment="benchmark",
    )
 
 
class GRAPEImputer:
    """
    GRAPE adapter conforming to the project's Imputer interface.
 
    Parameters
    ----------
    grape_root : str | None
        Path to DiffPuter/baselines/GRAPE. Auto-detected if None.
    epochs : int
        Training epochs. Paper default 20000 (slow but matches the
        published numbers). Lower (e.g. 5000) is fine for fast sweeps.
    device : str | None
        Torch device. None -> cuda if available else cpu.
    verbose : bool
        Print per-call summary.
 
    Hyperparameter overrides (advanced):
        node_dim, edge_dim, lr, known, valid, dropout, weight_decay,
        model_types, aggr, impute_hiddens, edge_mode, loss_mode.
        Any kwarg matching a field of the default args namespace is
        forwarded; unknown kwargs raise TypeError.
    """
 
    name = "GRAPE"
 
    _OVERRIDABLE = frozenset({
        "node_dim", "edge_dim", "lr", "known", "valid",
        "dropout", "weight_decay", "model_types", "aggr",
        "impute_hiddens", "edge_mode", "loss_mode",
        "opt_decay_step", "opt_decay_rate", "opt_scheduler",
        "gnn_activation", "impute_activation",
    })
 
    def __init__(
        self,
        grape_root: str | None = None,
        epochs: int = 20000,
        device: str | None = None,
        verbose: bool = False,
        **overrides,
    ):
        unknown = set(overrides) - self._OVERRIDABLE
        if unknown:
            raise TypeError(
                f"Unknown GRAPEImputer kwargs: {sorted(unknown)}. "
                f"Allowed: {sorted(self._OVERRIDABLE)}"
            )
        self.grape_root = grape_root
        self.epochs = epochs
        self._device_arg = device
        self.verbose = verbose
        self.overrides = overrides
 
    def _resolve_device(self) -> str:
        import torch
        if self._device_arg is not None:
            return self._device_arg
        return "cuda" if torch.cuda.is_available() else "cpu"
 
    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        """
        Impute missing values in X using GRAPE.
 
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Already-standardized data with NaN at missing positions.
        seed : int
            Random seed.
 
        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Fully observed array in the same scale as input X. Originally
            observed cells are preserved exactly; originally missing
            cells are filled by GRAPE.
        """
        _ensure_grape_on_path(self.grape_root)
 
        import torch
        from uci.uci_data import get_data_fix_mask  # type: ignore[import-not-found]
        from training.gnn_mdi import train_gnn_mdi  # type: ignore[import-not-found]
 
        device = self._resolve_device()
        torch.manual_seed(seed)
        np.random.seed(seed)
 
        from sklearn.preprocessing import MinMaxScaler
 
        observed_mask = ~np.isnan(X)  # True = observed (GRAPE convention)
        missing_mask = np.isnan(X)
 
        X_for_fit = X.copy()
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_inds = np.where(np.isnan(X_for_fit))
        X_for_fit[nan_inds] = np.take(col_means, nan_inds[1])
 
        mm = MinMaxScaler()
        X_mm = mm.fit_transform(X_for_fit).astype(np.float64)
 
        df_X = pd.DataFrame(X_mm)
 
        # Build args namespace and apply overrides
        args = _default_grape_args(seed=seed, epochs=self.epochs)
        for k, v in self.overrides.items():
            setattr(args, k, v)
 
        # Build the bipartite Data object with our fixed mask
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            data = get_data_fix_mask(df_X, observed_mask)
 
        # train_gnn_mdi writes logs to disk; isolate them in a tempdir
        with tempfile.TemporaryDirectory(prefix="grape_log_") as log_path:
            log_path_slash = log_path.rstrip("/") + "/"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = train_gnn_mdi(
                    data, args, log_path_slash, device, return_filled_X=True,
                )
 
        # Per GRAPE_benchmark.py line 227, the return is:
        #   X_filled, impute_model, model = train_gnn_mdi(...)
        if isinstance(result, tuple):
            X_filled = result[0]
        else:
            X_filled = result
 
        if isinstance(X_filled, torch.Tensor):
            X_filled = X_filled.detach().cpu().numpy()
        X_filled = np.asarray(X_filled, dtype=np.float64)
 
        if X_filled.shape != X.shape:
            raise RuntimeError(
                f"GRAPE returned shape {X_filled.shape}, expected {X.shape}. "
                "The fork's API may have changed; inspect train_gnn_mdi."
            )
 
        # Invert MinMax to return data in caller's original (standardised)
        # scale. Per GRAPE_benchmark.py line 230.
        X_filled_original_scale = mm.inverse_transform(X_filled)
 
        # Paranoia: preserve originally observed cells exactly
        result_arr = X_filled_original_scale.copy()
        result_arr[observed_mask] = X[observed_mask]
 
        if self.verbose:
            print(
                f"  [GRAPE] device={device} epochs={self.epochs} "
                f"missing={int(missing_mask.sum())}/{X.size}"
            )
 
        # GPU cleanup
        if device != "cpu":
            try:
                import torch as _torch
                _torch.cuda.empty_cache()
            except Exception:
                pass
 
        return result_arr