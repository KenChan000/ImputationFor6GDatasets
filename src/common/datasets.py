"""
Dataset registry.

Each `DatasetSpec` captures everything that differs between datasets, so the
tuning / imputation / downstream notebooks stay dataset-agnostic: set
`DATASET` at the top of a notebook, call `get_dataset(DATASET)`, and every
column list, file path, and result name follows from the spec.

To add a dataset: register a new `DatasetSpec` below. The hard requirements are
that `retained_cols` all exist in the loaded frame (ExperimentConfig.validate
enforces this) and that `group_col` is present in the loaded frame but NOT in
`retained_cols` (it rides along for the grouped split, never as a feature).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from .deepsense_data import load_deepsense_clean, BEAM_COLS


@dataclass(frozen=True)
class DatasetSpec:
    name: str                       # keys tuned_params.json, result filenames, cache dir
    data_dirname: str               # subfolder under <repo>/data
    raw_csv_name: str               # CSV inside the data dir
    loader: object                  # callable(data_dir) -> clean DataFrame
    target_cols: list               # columns to ampute (the beams)
    retained_cols: list             # full working matrix: targets + observed predictors
    mar_driver_cols: list           # observed column(s) driving MAR missingness
    downstream_feature_cols: list   # predictor inputs for SQ3 (GPS -> beam)
    group_col: str                  # sequence id for the grouped split (kept in frame, NOT retained)
    mar_score_mode: str = "single"

    @property
    def n_classes(self) -> int:
        """Beam codebook size = number of target columns."""
        return len(self.target_cols)


# ---------------------------------------------------------------------------
# Scenario 5
# ---------------------------------------------------------------------------
_S5_DROP = [
    "index", "unit1_rgb", "unit1_pwr_60ghz", "unit1_loc", "unit2_loc",
    "unit1_beam_index", "time_stamp[UTC]", "unit2_sat_used",
    "unit2_fix_type", "unit2_DGPS",
]
_S5_GPS = [
    "unit2_lat", "unit2_lon", "unit2_direction",
    "unit2_num_sat", "unit2_PDOP", "unit2_HDOP",
]

# ---------------------------------------------------------------------------
# Scenario 33
# ---------------------------------------------------------------------------
_S33_DROP = [
    "index", "unit1_rgb", "unit1_pwr_60ghz", "unit1_lidar", "unit1_radar",
    "unit1_loc", "unit2_loc", "unit1_beam", "unit1_max_pwr", "time_stamp",
    "unit2_spd_over_grnd_kmph", "unit2_altitude", "unit2_geo_sep",
    "unit2_mode_fix_type", "unit2_interpolated_position",
]
_S33_GPS = [
    "unit2_lat", "unit2_lon", "unit2_num_sats",
    "unit2_pdop", "unit2_hdop", "unit2_vdop",
]


REGISTRY: dict[str, DatasetSpec] = {
    "scenario5": DatasetSpec(
        name="scenario5",
        data_dirname="Scenario5",
        raw_csv_name="scenario5.csv",
        loader=partial(load_deepsense_clean,
                       csv_name="scenario5.csv",
                       beam_label_col="unit1_beam_index",
                       drop_columns=_S5_DROP),
        target_cols=BEAM_COLS,
        retained_cols=BEAM_COLS + _S5_GPS,
        mar_driver_cols=["unit2_HDOP"],
        downstream_feature_cols=["unit2_lat", "unit2_lon"],
        group_col="seq_index",
    ),
    "scenario33": DatasetSpec(
        name="scenario33",
        data_dirname="Scenario33",
        raw_csv_name="scenario33.csv",
        loader=partial(load_deepsense_clean,
                       csv_name="scenario33.csv",
                       beam_label_col="unit1_beam",
                       drop_columns=_S33_DROP),
        target_cols=BEAM_COLS,
        retained_cols=BEAM_COLS + _S33_GPS,
        mar_driver_cols=["unit2_hdop"],
        downstream_feature_cols=["unit2_lat", "unit2_lon"],
        group_col="seq_index",
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered: {sorted(REGISTRY)}"
        ) from None