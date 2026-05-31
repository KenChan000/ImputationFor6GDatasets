"""
MaMIMO (KU Leuven Ultra-Dense Indoor CSI) data loading + preprocessing.

Mirror of scenario5_data.py. Each measurement on disk is one complex CSI
matrix of shape (n_antennas, n_subcarriers); this module flattens every
sample into one row of a tabular DataFrame so the same amputation /
imputation / downstream notebooks can run on it unchanged.

Directory layout expected (matches the upload):

    <data_dir>/
        antenna_positions.npy            # (n_ant, 3) fixed array geometry
        samples/
            channel_measurement_000000.npy   # (n_ant, n_sub) complex
            channel_measurement_000001.npy
            ...
        positions.npy                    # (n_samples, 2 or 3) OPTIONAL user
                                         # coords — the downstream label

Design notes
------------
* Complex -> real: imputers are real-valued, so each complex CSI entry must
  become real column(s). Three options via `representation`:
      "magnitude" : |h|         -> n_ant*n_sub cols   (default; beam-power
                                    analogue, directly comparable to Scenario5)
      "power"     : |h|**2       -> n_ant*n_sub cols
      "reim"      : Re(h), Im(h) -> 2*n_ant*n_sub cols (lossless, keeps phase)
* Dimensionality: full magnitude is 64*100 = 6400 columns -- ~100x wider than
  Scenario5's 64 beams. MICE / HyperImpute / GRAPE / DiffPuter will be heavy at
  that width. Use `subcarriers` to subsample the frequency axis, or
  `aggregate="mean_subcarrier"` to collapse to n_ant features (one magnitude
  per antenna -> 64 columns, dimensionally identical to Scenario5).
* Order: samples are loaded in ascending file-index order and the parsed index
  is kept as the DataFrame INDEX (`sample_id`). That is the MaMIMO analogue of
  the Scenario5 timestamp -- it preserves and *proves* row order without adding
  a feature column. `.to_numpy()` ignores the index, so the experiment arrays
  stay feature-only while order/alignment remain recoverable.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_GLOB = "channel_measurement_*.npy"
_INDEX_RE = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# Per-sample parsing
# ---------------------------------------------------------------------------
def _sample_index(path: Path) -> int:
    """Parse the integer sample id out of a filename (...000123.npy -> 123)."""
    m = _INDEX_RE.findall(path.stem)
    return int(m[-1]) if m else -1


def csi_to_features(H: np.ndarray,
                    representation: str = "magnitude",
                    subcarriers=None,
                    aggregate=None) -> np.ndarray:
    """One complex CSI matrix (n_ant, n_sub) -> 1-D real feature vector.

    Flatten order is antenna-major (C-order): a0s0, a0s1, ..., a1s0, ...
    so it matches the column names produced by `feature_columns`.
    """
    if subcarriers is not None:
        H = H[:, list(subcarriers)]

    if aggregate == "mean_subcarrier":
        mag = np.abs(H)
        vec = (mag ** 2).mean(axis=1) if representation == "power" else mag.mean(axis=1)
        return vec.astype(float)

    if representation == "magnitude":
        return np.abs(H).ravel().astype(float)
    if representation == "power":
        return (np.abs(H) ** 2).ravel().astype(float)
    if representation == "reim":
        return np.concatenate([H.real.ravel(), H.imag.ravel()]).astype(float)
    raise ValueError(f"Unknown representation: {representation!r}")


def feature_columns(n_ant: int, n_sub: int,
                    representation: str = "magnitude",
                    subcarriers=None,
                    aggregate=None) -> list[str]:
    """Column names matching `csi_to_features` exactly (same order)."""
    subs = list(range(n_sub)) if subcarriers is None else list(subcarriers)

    if aggregate == "mean_subcarrier":
        tag = "pow" if representation == "power" else "mag"
        return [f"csi_{tag}_a{a:02d}" for a in range(n_ant)]

    if representation in ("magnitude", "power"):
        tag = "mag" if representation == "magnitude" else "pow"
        return [f"csi_{tag}_a{a:02d}_s{s:03d}" for a in range(n_ant) for s in subs]
    if representation == "reim":
        re_cols = [f"csi_re_a{a:02d}_s{s:03d}" for a in range(n_ant) for s in subs]
        im_cols = [f"csi_im_a{a:02d}_s{s:03d}" for a in range(n_ant) for s in subs]
        return re_cols + im_cols
    raise ValueError(f"Unknown representation: {representation!r}")


# ---------------------------------------------------------------------------
# Geometry + optional labels
# ---------------------------------------------------------------------------
def load_antenna_positions(data_dir, name: str = "antenna_positions.npy") -> np.ndarray:
    """Fixed array geometry, shape (n_ant, 3). Constant across samples."""
    return np.load(Path(data_dir) / name)


def load_user_positions(data_dir, name: str = "positions.npy"):
    """Per-sample user coordinates (downstream label), or None if absent.

    The uploaded set did not include these; localization downstream needs them.
    Returns an (n_samples, d) array aligned to ascending sample index, or None.
    """
    p = Path(data_dir) / name
    return np.load(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Main loader (mirrors load_scenario5_clean)
# ---------------------------------------------------------------------------
def load_mamimo_clean(data_dir,
                      samples_subdir: str = "samples",
                      representation: str = "magnitude",
                      subcarriers=None,
                      aggregate=None,
                      max_samples: int | None = None,
                      cache_path=None,
                      force: bool = False,
                      verbose: bool = True) -> pd.DataFrame:
    """Build one tabular DataFrame from a folder of complex CSI .npy samples.

    Returns a DataFrame whose INDEX is the parsed sample id (order anchor) and
    whose COLUMNS are the real-valued CSI features. NaN-filled row on a sample
    that fails to load or has an unexpected shape.
    """
    data_dir = Path(data_dir)

    if cache_path is not None and Path(cache_path).exists() and not force:
        if verbose:
            print(f"Loading cached clean data from {cache_path}")
        return pd.read_parquet(cache_path)

    sample_dir = data_dir / samples_subdir
    files = sorted(sample_dir.glob(SAMPLE_GLOB), key=_sample_index)
    if not files:
        raise FileNotFoundError(f"No '{SAMPLE_GLOB}' files under {sample_dir}")
    if max_samples is not None:
        files = files[:max_samples]

    # Infer geometry from the first sample.
    H0 = np.load(files[0])
    if H0.ndim != 2:
        raise ValueError(f"Expected 2-D CSI (n_ant, n_sub), got {H0.shape}")
    n_ant, n_sub = H0.shape
    cols = feature_columns(n_ant, n_sub, representation, subcarriers, aggregate)
    if verbose:
        print(f"{len(files)} samples | CSI per sample: {n_ant} ant x {n_sub} sub "
              f"({H0.dtype}) | representation={representation!r}"
              f"{f', subcarriers={list(subcarriers)}' if subcarriers is not None else ''}"
              f"{', aggregate=mean_subcarrier' if aggregate else ''} "
              f"-> {len(cols)} feature columns")

    rows = np.full((len(files), len(cols)), np.nan, dtype=float)
    ids = np.empty(len(files), dtype=int)
    n_fail = 0
    for i, f in enumerate(files):
        ids[i] = _sample_index(f)
        try:
            H = np.load(f)
            if H.shape != (n_ant, n_sub):
                n_fail += 1
                continue
            rows[i] = csi_to_features(H, representation, subcarriers, aggregate)
        except (OSError, ValueError):
            n_fail += 1

    clean = pd.DataFrame(rows, columns=cols, index=pd.Index(ids, name="sample_id"))

    if verbose:
        print(f"Clean data shape: {clean.shape}")
        if not clean.index.is_monotonic_increasing:
            print("  WARNING: sample_id index is not monotonic — check file naming.")
        if n_fail:
            print(f"  {n_fail} sample(s) failed to load -> NaN rows "
                  "(consider dropping before experiments)")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(cache_path)
        if verbose:
            print(f"Cached clean data to {cache_path}")

    return clean