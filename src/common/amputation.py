"""
Missingness mechanisms: MCAR, MAR, MNAR.
"""

from __future__ import annotations

import numpy as np

def _calibrate_intercept(scores: np.ndarray, slope: float, target_rate: float) -> float:
    flat = scores.reshape(-1)
    def mean_p(b):
        return float(np.mean(1.0 / (1.0 + np.exp(-(slope * flat + b)))))
    lo, hi = -50.0, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ampute_mcar(shape, prop, target_col_idx, rng):
    mask = np.zeros(shape, dtype=bool)
    sub_mask = rng.random((shape[0], len(target_col_idx))) < prop
    mask[:, target_col_idx] = sub_mask
    return mask


def _mar_row_score(data_scaled, driver_col_idx, score_mode):
    cols = data_scaled[:, driver_col_idx]
    if score_mode == "single":
        if cols.shape[1] != 1:
            raise ValueError(
                "score_mode='single' expects exactly one driver column, "
                f"got {cols.shape[1]}."
            )
        score = cols[:, 0]
    elif score_mode == "distance":
        centroid = cols.mean(axis=0)
        score = np.sqrt(((cols - centroid) ** 2).sum(axis=1))
    elif score_mode == "sum":
        score = cols.sum(axis=1)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode!r}")
    return (score - score.mean()) / (score.std() + 1e-8)


def ampute_mar(data_scaled, prop, target_col_idx, driver_col_idx, rng,
               score_mode="single"):
    n_rows = data_scaled.shape[0]
    mask = np.zeros(data_scaled.shape, dtype=bool)
    score = _mar_row_score(data_scaled, driver_col_idx, score_mode)
    slope = 1.0
    intercept = _calibrate_intercept(score, slope, prop)
    p_row = 1.0 / (1.0 + np.exp(-(slope * score + intercept)))
    draws = rng.random((n_rows, len(target_col_idx)))
    sub_mask = draws < p_row[:, None]
    mask[:, target_col_idx] = sub_mask
    return mask


def ampute_mnar(data_scaled, prop, target_col_idx, rng):
    mask = np.zeros(data_scaled.shape, dtype=bool)
    cell_vals = data_scaled[:, target_col_idx]
    slope = 1.0
    intercept = _calibrate_intercept(cell_vals, slope, prop)
    p_cell = 1.0 / (1.0 + np.exp(-(slope * cell_vals + intercept)))
    draws = rng.random(cell_vals.shape)
    sub_mask = draws < p_cell
    mask[:, target_col_idx] = sub_mask
    return mask


def build_mask(mechanism, data_scaled, prop, target_col_idx, rng,
               driver_col_idx=None, mar_score_mode="single"):
    m = mechanism.lower()
    if m == "mcar":
        return ampute_mcar(data_scaled.shape, prop, target_col_idx, rng)
    if m == "mar":
        if driver_col_idx is None:
            raise ValueError("MAR requires driver_col_idx")
        return ampute_mar(data_scaled, prop, target_col_idx, driver_col_idx,
                          rng, score_mode=mar_score_mode)
    if m == "mnar":
        return ampute_mnar(data_scaled, prop, target_col_idx, rng)
    raise ValueError(f"Unknown mechanism: {mechanism}")