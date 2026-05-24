"""
Hyperparameter tuning for imputation methods.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .amputation import build_mask


def _fill_residual_nans(arr, masked):
    if not np.isnan(arr).any():
        return arr
    col_means = np.nanmean(masked, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_pos = np.isnan(arr)
    arr[nan_pos] = np.take(col_means, np.where(nan_pos)[1])
    return arr


def tune_imputer(clean_data: pd.DataFrame,
                 cfg,
                 factory,
                 param_grid,
                 param_name: str = "param",
                 prop: float = 0.30,
                 tuning_seeds=(0, 1, 2),
                 verbose: bool = True) -> dict:
    data_scaled = cfg.scale(clean_data)
    target_idx = cfg.target_idx

    masked_sets = []
    for s in tuning_seeds:
        rng = np.random.default_rng(s)
        mask = build_mask("mcar", data_scaled, prop, target_idx, rng)
        dm = data_scaled.copy()
        dm[mask] = np.nan
        masked_sets.append((mask, dm, data_scaled[mask]))

    records = []
    for value in param_grid:
        rmses = []
        for seed, (mask, dm, truth) in zip(tuning_seeds, masked_sets):
            imputer = factory(value)
            imputed = imputer.fit_transform(dm.copy(), seed=seed)
            imputed = _fill_residual_nans(imputed, dm)
            err = truth - imputed[mask]
            rmses.append(float(np.sqrt(np.mean(err ** 2))))
        rec = {
            param_name: value,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
        }
        records.append(rec)
        if verbose:
            print(f"  {param_name}={value}: "
                  f"RMSE {rec['rmse_mean']:.4f} ± {rec['rmse_std']:.4f}")

    table = pd.DataFrame(records).sort_values("rmse_mean").reset_index(drop=True)
    best = table.iloc[0]
    if verbose:
        print(f"  -> best {param_name} = {best[param_name]} "
              f"(RMSE {best['rmse_mean']:.4f})")
    return {
        "best_value": best[param_name],
        "best_rmse": float(best["rmse_mean"]),
        "table": table,
    }

def tune_knn(clean_data, cfg, knn_class,
             grid=(3, 5, 10, 20, 50), **kw) -> dict:
    return tune_imputer(
        clean_data, cfg,
        factory=lambda k: knn_class(k=k),
        param_grid=grid, param_name="n_neighbors", **kw,
    )


def softimpute_shrinkage_grid(clean_data, cfg, fractions=(100, 50, 20, 10),
                              include_default=True):
    data_scaled = cfg.scale(clean_data)
    filled = np.where(np.isnan(data_scaled),
                      np.nanmean(data_scaled, axis=0), data_scaled)
    max_sv = float(np.linalg.svd(filled, compute_uv=False)[0])
    grid = [round(max_sv / d, 4) for d in fractions]
    if include_default:
        grid = [None] + grid
    print(f"max singular value = {max_sv:.3f} "
          f"(fancyimpute default shrinkage = {max_sv/50:.4f})")
    return grid


def tune_softimpute(clean_data, cfg, softimpute_class, grid, **kw) -> dict:
    return tune_imputer(
        clean_data, cfg,
        factory=lambda s: softimpute_class(shrinkage_value=s),  # max_rank stays None
        param_grid=grid, param_name="shrinkage_value", **kw,
    )

import json
from pathlib import Path


def save_tuned_params(path, dataset: str, params: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = {}
    if path.is_file():
        store = json.loads(path.read_text())
    store.setdefault(dataset, {})
    store[dataset].update(params)
    path.write_text(json.dumps(store, indent=2))


def load_tuned_params(path, dataset: str) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get(dataset, {})