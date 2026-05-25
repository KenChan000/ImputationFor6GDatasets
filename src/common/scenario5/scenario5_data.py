"""
Scenario 5 (DeepSense) data loading + preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_BEAMS = 64
BEAM_COLS = [f"beam_{i:02d}" for i in range(N_BEAMS)]

DROP_COLUMNS = [
    "index", "unit1_rgb", "unit1_pwr_60ghz", "unit1_loc", "unit2_loc",
    "unit1_beam_index", "seq_index", "time_stamp[UTC]", "unit2_sat_used",
    "unit2_fix_type", "unit2_DGPS",
]

def parse_loc_file(rel_path, base_dir: Path):
    """Read a location .txt file -> (lat, lon). NaNs on failure."""
    if pd.isna(rel_path):
        return np.nan, np.nan
    try:
        full = base_dir / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        content = Path(full).read_text().strip()
        parts = [p.strip() for p in content.replace(",", " ").split()]
        return float(parts[0]), float(parts[1])
    except (FileNotFoundError, ValueError, IndexError):
        return np.nan, np.nan


def parse_pwr_file(rel_path, base_dir: Path):
    """Read an mmWave power .txt file -> length-64 array. NaNs on failure."""
    if pd.isna(rel_path):
        return np.full(N_BEAMS, np.nan)
    try:
        full = base_dir / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        vals = np.loadtxt(full)
        if vals.shape != (N_BEAMS,):
            return np.full(N_BEAMS, np.nan)
        return vals
    except (FileNotFoundError, ValueError):
        return np.full(N_BEAMS, np.nan)


def load_scenario5_clean(data_dir,
                         csv_name: str = "scenario5.csv",
                         cache_path=None,
                         force: bool = False,
                         verbose: bool = True) -> pd.DataFrame:
    data_dir = Path(data_dir)

    if cache_path is not None and Path(cache_path).exists() and not force:
        if verbose:
            print(f"Loading cached clean data from {cache_path}")
        return pd.read_parquet(cache_path)

    base_dir = data_dir

    # --- load index CSV, drop unnamed columns ---
    df = pd.read_csv(data_dir / csv_name)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # --- parse mobile-receiver GPS coordinates ---
    if verbose:
        print("Parsing location files...")
    df[["unit2_lat", "unit2_lon"]] = df["unit2_loc"].apply(
        lambda p: pd.Series(parse_loc_file(p, base_dir))
    )

    # --- parse 64-beam mmWave power files ---
    if verbose:
        print("Parsing mmWave power files...")
    pwr_matrix = np.vstack([parse_pwr_file(p, base_dir) for p in df["unit1_pwr_60ghz"]])
    pwr_df = pd.DataFrame(pwr_matrix, columns=BEAM_COLS, index=df.index)

    # sanity check: parsed argmax vs recorded optimal beam index
    if "unit1_beam_index" in df.columns:
        parsed_argmax = np.argmax(pwr_matrix, axis=1) + 1
        match = (parsed_argmax == df["unit1_beam_index"].values).mean()
        if verbose:
            print(f"Argmax matches unit1_beam_index: {match * 100:.2f}%")

    # --- assemble and drop the redundant / leaky / constant columns ---
    df_add_pwr = pd.concat([pwr_df, df], axis=1)
    clean = df_add_pwr.drop(columns=[c for c in DROP_COLUMNS if c in df_add_pwr.columns])

    if verbose:
        print(f"Clean data shape: {clean.shape}")
        n_missing = int(clean.isna().sum().sum())
        print(f"Missing values after parsing: {n_missing}")
        if n_missing:
            print("  (rows with parse failures — consider dropping before experiments)")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(cache_path)
        if verbose:
            print(f"Cached clean data to {cache_path}")

    return clean