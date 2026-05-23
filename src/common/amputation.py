"""
Missingness mechanisms: MCAR, MAR, MNAR.

"""

from __future__ import annotations

import numpy as np

def _calibrate_intercept(scores: np.ndarray, slope: float, target_rate: float) -> float:
    """Bisection-solve for intercept b so mean(sigmoid(slope*scores + b)) == target_rate.

    Used by MAR (scores are row-level driver sums) and MNAR (scores are
    per-cell values, flattened). Works for any input shape since we only
    need the global mean.
    """
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


def ampute_mcar(shape: tuple[int, int], prop: float,
                target_col_idx: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    sub_mask = rng.random((shape[0], len(target_col_idx))) < prop
    mask[:, target_col_idx] = sub_mask
    return mask


def ampute_mar(data_scaled: np.ndarray, prop: float,
               target_col_idx: np.ndarray,
               driver_col_idx: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    n_rows = data_scaled.shape[0]
    mask = np.zeros(data_scaled.shape, dtype=bool)

    # Row-level score from standardised driver columns
    score = data_scaled[:, driver_col_idx].sum(axis=1)
    score = (score - score.mean()) / (score.std() + 1e-8)

    slope = 1.0
    intercept = _calibrate_intercept(score, slope, prop)
    p_row = 1.0 / (1.0 + np.exp(-(slope * score + intercept)))  # shape (n_rows,)

    draws = rng.random((n_rows, len(target_col_idx)))
    sub_mask = draws < p_row[:, None]
    mask[:, target_col_idx] = sub_mask
    return mask


def ampute_mnar(data_scaled: np.ndarray, prop: float,
                target_col_idx: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(data_scaled.shape, dtype=bool)

    # Per-cell scores: just the standardised target values themselves.
    cell_vals = data_scaled[:, target_col_idx]  # shape (n_rows, n_targets)

    slope = 1.0  # high values -> high miss probability
    intercept = _calibrate_intercept(cell_vals, slope, prop)
    p_cell = 1.0 / (1.0 + np.exp(-(slope * cell_vals + intercept)))

    draws = rng.random(cell_vals.shape)
    sub_mask = draws < p_cell
    mask[:, target_col_idx] = sub_mask
    return mask


# Mechanism registry so drivers can dispatch by name without an if/elif ladder.
def build_mask(mechanism: str, data_scaled: np.ndarray, prop: float,
               target_col_idx: np.ndarray, rng: np.random.Generator,
               driver_col_idx: np.ndarray | None = None) -> np.ndarray:
    """Single entry point. `mechanism` in {'mcar', 'mar', 'mnar'}."""
    m = mechanism.lower()
    if m == "mcar":
        return ampute_mcar(data_scaled.shape, prop, target_col_idx, rng)
    if m == "mar":
        if driver_col_idx is None:
            raise ValueError("MAR requires driver_col_idx")
        return ampute_mar(data_scaled, prop, target_col_idx, driver_col_idx, rng)
    if m == "mnar":
        return ampute_mnar(data_scaled, prop, target_col_idx, rng)
    raise ValueError(f"Unknown mechanism: {mechanism}")