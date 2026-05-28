"""
Optuna-based hyperparameter tuning for imputation methods.
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


def _build_masked_sets(cfg, clean_data, prop, tuning_seeds):
    """Identical to tune_imputer: fixed MCAR masks, one per tuning seed."""
    data_scaled = cfg.scale(clean_data)
    target_idx = cfg.target_idx
    masked_sets = []
    for s in tuning_seeds:
        rng = np.random.default_rng(s)
        mask = build_mask("mcar", data_scaled, prop, target_idx, rng)
        dm = data_scaled.copy()
        dm[mask] = np.nan
        masked_sets.append((mask, dm, data_scaled[mask]))
    return masked_sets


def _mean_rmse(factory_kwargs, factory, masked_sets, tuning_seeds):
    """Mean RMSE over the fixed masked sets — matches tune_imputer's metric."""
    rmses = []
    for seed, (mask, dm, truth) in zip(tuning_seeds, masked_sets):
        imputer = factory(**factory_kwargs)
        imputed = imputer.fit_transform(dm.copy(), seed=seed)
        imputed = _fill_residual_nans(imputed, dm)
        err = truth - imputed[mask]
        rmses.append(float(np.sqrt(np.mean(err ** 2))))
    return float(np.mean(rmses)), float(np.std(rmses))


def tune_imputer_optuna(clean_data, cfg, factory, search_space,
                        prop: float = 0.30,
                        tuning_seeds=(0,),
                        n_trials: int = 20,
                        sampler_seed: int = 0,
                        direction: str = "minimize",
                        complexity_key: str | None = None,
                        complexity_weight: float = 0.0,
                        verbose: bool = True) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    masked_sets = _build_masked_sets(cfg, clean_data, prop, tuning_seeds)

    def _suggest(trial, name, spec):
        kind = spec[0]
        if kind == "categorical":
            return trial.suggest_categorical(name, spec[1])
        if kind == "int":
            step = spec[3] if len(spec) > 3 else 1
            return trial.suggest_int(name, spec[1], spec[2], step=step)
        if kind == "loguniform":
            return trial.suggest_float(name, spec[1], spec[2], log=True)
        raise ValueError(f"Unknown search spec for {name!r}: {spec}")

    rows = []

    def objective(trial):
        params = {n: _suggest(trial, n, s) for n, s in search_space.items()}
        rmse_mean, rmse_std = _mean_rmse(params, factory, masked_sets, tuning_seeds)
        rows.append({**params, "rmse_mean": rmse_mean, "rmse_std": rmse_std})
        score = rmse_mean
        if complexity_key is not None and complexity_weight:
            score = score + complexity_weight * float(np.log2(params[complexity_key]))
        if verbose:
            pstr = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  trial {trial.number:2d}: {pstr} -> "
                  f"RMSE {rmse_mean:.4f} ± {rmse_std:.4f}")
        return score

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    table = (pd.DataFrame(rows)
             .sort_values("rmse_mean")
             .reset_index(drop=True))
    best = study.best_params
    # best_rmse is the pure-RMSE of the winning params (not the penalised score)
    best_row = table.iloc[0]
    if verbose:
        print(f"  -> best {best} (RMSE {best_row['rmse_mean']:.4f})")
    return {
        "best_params": best,
        "best_rmse": float(best_row["rmse_mean"]),
        "study": study,
        "table": table,
    }
