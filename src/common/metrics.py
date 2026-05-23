"""
Imputation-quality metrics: point-wise (RMSE, MAE), distributional
(Wasserstein-1), moment-based (mean shift, variance retention), and joint
dependency (correlation Frobenius distance).

Scenario-agnostic: everything takes arrays + a boolean mask. No globals.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def _per_column_distribution_metrics(data_truth: np.ndarray,
                                     data_imputed: np.ndarray,
                                     mask: np.ndarray) -> dict:
    """Per-column distribution + moment metrics, aggregated across columns
    (mean and std). Only considers columns with at least 2 masked cells.
    """
    n_cols = data_truth.shape[1]
    w1_per_col            = []
    mean_shift_per_col    = []
    var_retention_per_col = []

    for j in range(n_cols):
        cell_mask = mask[:, j]
        if int(cell_mask.sum()) < 2:
            continue

        truth_j = data_truth[cell_mask, j]
        imp_j   = data_imputed[cell_mask, j]

        w1_per_col.append(float(wasserstein_distance(truth_j, imp_j)))
        mean_shift_per_col.append(float(abs(imp_j.mean() - truth_j.mean())))

        var_truth = float(truth_j.var())
        if var_truth > 1e-12:
            var_retention_per_col.append(float(imp_j.var() / var_truth))

    if not w1_per_col:
        return {
            "wasserstein_mean":        float("nan"),
            "wasserstein_std":         float("nan"),
            "mean_shift_mean":         float("nan"),
            "mean_shift_std":          float("nan"),
            "variance_retention_mean": float("nan"),
            "variance_retention_std":  float("nan"),
            "n_cols_with_masking":     0,
        }

    return {
        "wasserstein_mean":        float(np.mean(w1_per_col)),
        "wasserstein_std":         float(np.std(w1_per_col)),
        "mean_shift_mean":         float(np.mean(mean_shift_per_col)),
        "mean_shift_std":          float(np.std(mean_shift_per_col)),
        "variance_retention_mean": (float(np.mean(var_retention_per_col))
                                    if var_retention_per_col else float("nan")),
        "variance_retention_std":  (float(np.std(var_retention_per_col))
                                    if var_retention_per_col else float("nan")),
        "n_cols_with_masking":     len(w1_per_col),
    }


def _correlation_frobenius_distance(data_truth: np.ndarray,
                                    data_imputed: np.ndarray) -> float:
    """Frobenius distance between truth and imputed correlation matrices.
    Lower is better; zero means correlations are perfectly preserved.
    """
    corr_truth = np.corrcoef(data_truth,   rowvar=False)
    corr_imp   = np.corrcoef(data_imputed, rowvar=False)
    corr_truth = np.nan_to_num(corr_truth, nan=0.0)
    corr_imp   = np.nan_to_num(corr_imp,   nan=0.0)
    return float(np.linalg.norm(corr_truth - corr_imp, ord="fro"))


def _fill_residual_nans(data_imputed: np.ndarray,
                        data_masked: np.ndarray) -> np.ndarray:
    """Fallback for imputers that occasionally leave NaNs: column-mean fill."""
    if not np.isnan(data_imputed).any():
        return data_imputed
    col_means = np.nanmean(data_masked, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    out = data_imputed.copy()
    nan_pos = np.isnan(out)
    out[nan_pos] = np.take(col_means, np.where(nan_pos)[1])
    return out


def evaluate_one_run(data_scaled: np.ndarray, mask: np.ndarray,
                     imputer, seed: int) -> dict:
    """Apply mask -> impute -> compute the full metric set.

    `imputer` is any object exposing .fit_transform(array, seed=...) and .name,
    matching the imputers.imputers.Imputer interface.
    """
    data_masked = data_scaled.copy()
    data_masked[mask] = np.nan

    data_imputed = imputer.fit_transform(data_masked, seed=seed)
    data_imputed = _fill_residual_nans(data_imputed, data_masked)

    err = data_scaled[mask] - data_imputed[mask]
    rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")
    mae  = float(np.mean(np.abs(err)))       if err.size else float("nan")

    dist_metrics = _per_column_distribution_metrics(data_scaled, data_imputed, mask)
    corr_frob = _correlation_frobenius_distance(data_scaled, data_imputed)

    return {
        "rmse": rmse,
        "mae":  mae,
        **dist_metrics,
        "corr_frob":      corr_frob,
        "n_masked_cells": int(mask.sum()),
    }