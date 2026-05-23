"""
Experiment configuration + generic drivers for imputation experiments.

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .amputation import build_mask
from .metrics import evaluate_one_run


@dataclass
class ExperimentConfig:
    """
    target_cols  : columns to ampute.
    retained_cols: full set of columns kept in the working matrix.
    mar_driver_cols: observed columns that drive MAR missingness.
    proportions  : missing fractions to sweep.
    n_seeds      : repetitions per cell.
    scenarios    : (display_name, mechanism) pairs to run.
    """
    target_cols: list[str]
    retained_cols: list[str]
    mar_driver_cols: list[str]
    proportions: list[float] = field(default_factory=lambda: [0.10, 0.30, 0.50])
    n_seeds: int = 1
    scenarios: list[tuple[str, str]] = field(default_factory=lambda: [
        ("MCAR", "mcar"), ("MAR", "mar"), ("MNAR", "mnar"),
    ])

    def col_index(self, cols: list[str]) -> np.ndarray:
        lut = {c: i for i, c in enumerate(self.retained_cols)}
        return np.array([lut[c] for c in cols])

    @property
    def target_idx(self) -> np.ndarray:
        return self.col_index(self.target_cols)

    @property
    def mar_driver_idx(self) -> np.ndarray:
        return self.col_index(self.mar_driver_cols)

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.retained_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"clean_data is missing {len(missing)} expected column(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if df[self.retained_cols].isna().any().any():
            raise ValueError(
                "clean_data already contains NaN values in retained columns. "
                "Amputation experiments require fully observed input."
            )

    def scale(self, df: pd.DataFrame) -> np.ndarray:
        data_full = df[self.retained_cols].to_numpy(dtype=float)
        return StandardScaler().fit_transform(data_full)


def _summarise_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse", "mae",
        "wasserstein_mean", "wasserstein_std",
        "mean_shift_mean",  "mean_shift_std",
        "variance_retention_mean", "variance_retention_std",
        "corr_frob",
    ]
    agg_spec = {}
    for m in metric_cols:
        agg_spec[f"{m}_seedmean"] = (m, "mean")
        agg_spec[f"{m}_seedstd"]  = (m, "std")
    agg_spec["avg_masked_cells"] = ("n_masked_cells", "mean")
    agg_spec["n_seeds"] = ("seed", "count")

    return (
        results_df
        .groupby(["scenario", "proportion", "method"], sort=False)
        .agg(**agg_spec)
        .round(4)
        .reset_index()
    )


def run_experiments(clean_data: pd.DataFrame,
                    config: ExperimentConfig,
                    imputers: list,
                    scenario_filter: list[str] | None = None,
                    verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all (scenario x proportion x method x seed) combinations.

    Identical logic to the notebook version, but driven by `config` instead
    of globals so it works for any column layout.
    """
    config.validate(clean_data)
    data_scaled = config.scale(clean_data)

    target_idx = config.target_idx
    mar_driver_idx = config.mar_driver_idx

    scenarios = config.scenarios

    if scenario_filter is not None:
        scenarios = [s for s in scenarios if s[0] in scenario_filter]
        if verbose:
            print(f"Filtered to {len(scenarios)} scenario(s): "
                  f"{[s[0] for s in scenarios]}")

    total_runs = (len(scenarios) * len(config.proportions)
                  * len(imputers) * config.n_seeds)
    if verbose:
        print(f"Total runs: {total_runs} "
              f"({len(scenarios)} scenarios x {len(config.proportions)} props "
              f"x {len(imputers)} methods x {config.n_seeds} seeds)")

    records = []
    run_counter = 0
    t_start = time.time()

    for scenario_name, mechanism in scenarios:
        for prop in config.proportions:
            for imputer in imputers:
                method_t0 = time.time()
                for seed in range(config.n_seeds):
                    rng = np.random.default_rng(seed)
                    mask = build_mask(mechanism, data_scaled, prop,
                                      target_idx, rng,
                                      driver_col_idx=mar_driver_idx)
                    metrics = evaluate_one_run(data_scaled, mask, imputer, seed)
                    records.append({
                        "scenario": scenario_name,
                        "proportion": prop,
                        "method": imputer.name,
                        "seed": seed,
                        **metrics,
                    })
                    run_counter += 1

                if verbose:
                    elapsed = time.time() - method_t0
                    print(f"  [{run_counter}/{total_runs}] "
                          f"{scenario_name} | p={prop} | {imputer.name}: "
                          f"{elapsed:.1f}s ({elapsed/config.n_seeds:.2f}s/seed)")

    if verbose:
        print(f"\nTotal runtime: {time.time() - t_start:.1f}s")

    results_df = pd.DataFrame(records)
    summary_df = _summarise_results(results_df)
    return results_df, summary_df