"""
Imputation-benchmark results explorer (Streamlit).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RECON_SUFFIX = "_imputation_results.csv"
DOWN_SUFFIX = "_downstream_full_results.csv"

CANON = {
    "mean": "Mean", "knn": "kNN", "mice": "MICE", "softimpute": "SoftImpute",
    "hyperimpute": "HyperImpute", "grape": "GRAPE", "diffputer": "DiffPuter",
    "clean": "Clean",
}
METHOD_ORDER = ["Mean", "kNN", "MICE", "SoftImpute", "HyperImpute",
                "GRAPE", "DiffPuter", "Clean"]
MECH_ORDER = ["MCAR", "MAR", "MNAR", "MCAR-Row"]

METHOD_PLOT_ORDER = ["Mean", "kNN", "MICE", "HyperImpute",
                     "SoftImpute", "GRAPE", "DiffPuter"]
METHOD_COLORS = {
    "Mean": "#1f77b4", "kNN": "#ff7f0e", "MICE": "#2ca02c",
    "HyperImpute": "#d62728", "SoftImpute": "#9467bd",
    "GRAPE": "#8c564b", "DiffPuter": "#e377c2", "Clean": "#333333",
}

RECON_METRICS = {
    "RMSE": ("rmse", True),
    "MAE": ("mae", True),
    "Wasserstein-1D": ("wasserstein_mean", True),
    "Mean shift": ("mean_shift_mean", True),
    "Variance retention": ("variance_retention_mean", False),
    "Covariance Frobenius": ("corr_frob", True),
}
DOWN_METRICS = {
    f"Top-{k} accuracy": (f"top{k}_acc", f"top{k}_acc_std", False)
    for k in (1, 2, 3, 4, 5)
}



def find_root() -> Path:
    """Walk up from this file and the CWD looking for a `results` folder."""
    here = Path(__file__).resolve()
    for c in [*here.parents, Path.cwd(), *Path.cwd().parents]:
        if (c / "results" / "csv").is_dir() or (c / "results" / "tuned_params.json").is_file():
            return c
    return here.parents[2] if len(here.parents) > 2 else Path.cwd()


def canon(name) -> str:
    return CANON.get(str(name).strip().lower(), str(name))


@st.cache_data(show_spinner=False)
def discover_datasets(csv_dir_str: str) -> dict:
    """{dataset: {'recon': path|None, 'down': path|None}} from the CSV filenames."""
    csv_dir = Path(csv_dir_str)
    out: dict = {}
    if not csv_dir.is_dir():
        return out
    for f in sorted(csv_dir.glob("*.csv")):
        for suffix, key in ((RECON_SUFFIX, "recon"), (DOWN_SUFFIX, "down")):
            if f.name.endswith(suffix):
                ds = f.name[: -len(suffix)]
                out.setdefault(ds, {"recon": None, "down": None})[key] = str(f)
    return out


@st.cache_data(show_spinner=False)
def load_results(path_str: str) -> pd.DataFrame:
    """Read + normalise one results CSV (column names, method casing, proportion)."""
    df = pd.read_csv(path_str)
    if "scenario" in df.columns and "mechanism" not in df.columns:
        df = df.rename(columns={"scenario": "mechanism"})
    df["method"] = df["method"].map(canon)
    df["proportion"] = df["proportion"].astype(float)
    df["prop_label"] = (df["proportion"] * 100).round().astype(int).astype(str) + "%"
    return df


@st.cache_data(show_spinner=False)
def load_tuned(path_str: str) -> dict:
    p = Path(path_str)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def pick_multi(label, options, key, default=None):
    options = list(options)
    default = options if default is None else [d for d in default if d in options]
    if hasattr(st, "pills"):
        sel = st.pills(label, options, selection_mode="multi", default=default, key=key)
    else:
        sel = st.multiselect(label, options, default=default, key=key)
    return list(sel or [])


def ordered(values, order):
    """Keep only present values, in the canonical order, then any extras."""
    present = [v for v in order if v in set(values)]
    extras = [v for v in values if v not in order]
    return present + sorted(extras)


def build_plot_df(df: pd.DataFrame, view: str, metric_label: str,
                  seeds: list | None) -> pd.DataFrame:
    """Return columns: method, mechanism, prop_label, proportion, value, lower, upper."""
    if view == "recon":
        col, _ = RECON_METRICS[metric_label]
        sub = df if seeds is None else df[df["seed"].isin(seeds)]
        g = (sub.groupby(["mechanism", "proportion", "prop_label", "method"])[col]
                .agg(value="mean", sd="std").reset_index())
        g["sd"] = g["sd"].fillna(0.0)
    else:
        vcol, scol, _ = DOWN_METRICS[metric_label]
        g = df[["mechanism", "proportion", "prop_label", "method", vcol, scol]].copy()
        g = g.rename(columns={vcol: "value", scol: "sd"})
        g["sd"] = g["sd"].fillna(0.0)

    g["lower"] = (g["value"] - g["sd"]).clip(lower=0)
    g["upper"] = g["value"] + g["sd"]
    return g


def legend_html(methods: list, show_clean: bool) -> str:
    """One compact shared legend (the per-panel charts hide their own)."""
    chips = []
    for m in methods:
        chips.append(
            f'<span style="display:inline-flex;align-items:center;margin:0 14px 4px 0;'
            f'white-space:nowrap;font-size:0.85rem;">'
            f'<span style="width:12px;height:12px;background:{METHOD_COLORS[m]};'
            f'display:inline-block;margin-right:5px;border-radius:2px;"></span>{m}</span>'
        )
    if show_clean:
        chips.append(
            f'<span style="display:inline-flex;align-items:center;margin:0 14px 4px 0;'
            f'white-space:nowrap;font-size:0.85rem;">'
            f'<span style="width:16px;border-top:2px dashed {METHOD_COLORS["Clean"]};'
            f'display:inline-block;margin-right:5px;"></span>Clean (ceiling)</span>'
        )
    return '<div style="margin:2px 0 8px 0;">' + "".join(chips) + "</div>"


def make_panel(df_mech: pd.DataFrame, metric_label: str, mechanism: str,
               y_max: float) -> alt.Chart:
    """Single mechanism: x = proportion, colour = method, grouped via xOffset.
    Error bars are a rule (+ tick cap) so they centre on each bar; `Clean`
    becomes a dashed ceiling line. No facet -> the chart fills its column."""
    prop_dom = sorted(df_mech["prop_label"].unique(), key=lambda s: int(s[:-1]))
    has_clean = "Clean" in set(df_mech["method"])
    bars_df = df_mech[df_mech["method"] != "Clean"] if has_clean else df_mech
    methods = [m for m in METHOD_PLOT_ORDER if m in set(bars_df["method"])]
    color_scale = alt.Scale(domain=methods, range=[METHOD_COLORS[m] for m in methods])
    y_scale = alt.Scale(domain=[0, y_max])

    base = alt.Chart(bars_df).encode(
        x=alt.X("prop_label:N", sort=prop_dom, title="Missingness proportion"),
        xOffset=alt.XOffset("method:N", sort=methods),
    )
    bars = base.mark_bar().encode(
        y=alt.Y("value:Q", title=metric_label, scale=y_scale),
        color=alt.Color("method:N", sort=methods, scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("method:N", title="Method"),
            alt.Tooltip("prop_label:N", title="Proportion"),
            alt.Tooltip("value:Q", title=metric_label, format=".4f"),
            alt.Tooltip("sd:Q", title="± std", format=".4f"),
        ],
    )
    errs = base.mark_rule(color="#333", size=1).encode(y="lower:Q", y2="upper:Q")
    caps = base.mark_tick(color="#333", thickness=1, size=5).encode(y="upper:Q")
    layers = [bars, errs, caps]

    if has_clean:
        clean_val = float(df_mech.loc[df_mech["method"] == "Clean", "value"].mean())
        rule = (alt.Chart(pd.DataFrame({"y": [clean_val]}))
                .mark_rule(strokeDash=[4, 3], color=METHOD_COLORS["Clean"], size=1.3)
                .encode(y="y:Q"))
        layers.append(rule)

    return alt.layer(*layers).properties(height=280, title=mechanism)


def render_tuned(tuned: dict, dataset: str):
    if dataset not in tuned:
        st.info(f"No tuned parameters found for **{dataset}** in tuned_params.json.")
        return
    params = tuned[dataset]
    scalars = {k: v for k, v in params.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in params.items() if isinstance(v, dict)}

    if scalars:
        sdf = pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in scalars.items()]
        )
        st.dataframe(sdf, hide_index=True, use_container_width=True)
    for name, block in nested.items():
        with st.expander(f"`{name}` hyperparameters", expanded=True):
            bdf = pd.DataFrame(
                [{"parameter": k, "value": v} for k, v in block.items()]
            )
            st.dataframe(bdf, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Imputation benchmark explorer",
                       layout="wide", page_icon="📊")
    st.title("📊 Imputation benchmark explorer")

    default_root = str(find_root())
    with st.sidebar:
        st.header("Source")
        root = Path(st.text_input("Repo root", value=default_root,
                                  help="Folder that contains results/")).expanduser()
        csv_dir = root / "results" / "csv"
        tuned_path = root / "results" / "tuned_params.json"

    datasets = discover_datasets(str(csv_dir))
    if not datasets:
        st.error(f"No result CSVs found in `{csv_dir}`. Expected files ending in "
                 f"`{RECON_SUFFIX}` or `{DOWN_SUFFIX}`.")
        st.stop()
    tuned = load_tuned(str(tuned_path))

    with st.sidebar:
        st.header("Filters")
        dataset = st.selectbox("Dataset", sorted(datasets), key="dataset")

        view_label = st.radio(
            "View",
            ["Reconstruction (SQ1/SQ2)", "Downstream (SQ3)"],
            key="view",
        )
        view = "recon" if view_label.startswith("Reconstruction") else "down"

        path = datasets[dataset]["recon" if view == "recon" else "down"]
        if path is None:
            kind = "reconstruction" if view == "recon" else "downstream"
            st.warning(f"No {kind} file for **{dataset}** yet.")
            st.stop()

    df = load_results(path)

    with st.sidebar:
        metric_dict = RECON_METRICS if view == "recon" else DOWN_METRICS
        metric_label = st.selectbox("Metric", list(metric_dict), key=f"metric_{view}")

        mech_sel = pick_multi("Mechanism",
                              ordered(df["mechanism"].unique(), MECH_ORDER),
                              key=f"mech_{view}")
        prop_opts = sorted(df["proportion"].unique())
        prop_labels = pick_multi("Proportion", [f"{int(p*100)}%" for p in prop_opts],
                                 key=f"prop_{view}")
        sel_props = [p for p in prop_opts if f"{int(p*100)}%" in prop_labels]

        method_sel = pick_multi("Method",
                                ordered(df["method"].unique(), METHOD_ORDER),
                                key=f"method_{view}")

        if view == "recon" and "seed" in df.columns:
            seed_opts = sorted(df["seed"].unique())
            seeds = pick_multi("Seed", seed_opts, key="seed")
            seeds = seeds or None
        else:
            seeds = None
            st.caption("Seed filter applies to the reconstruction view only — "
                       "downstream results are aggregated over seeds.")


    st.subheader(f"Tuned hyperparameters — {dataset}")
    render_tuned(tuned, dataset)
    st.divider()

    fdf = df[
        df["mechanism"].isin(mech_sel)
        & df["proportion"].isin(sel_props)
        & df["method"].isin(method_sel)
    ]
    if view == "recon" and seeds is not None:
        fdf = fdf[fdf["seed"].isin(seeds)]

    st.subheader(f"{view_label} — {metric_label}")
    if fdf.empty:
        st.warning("No rows match the current filters. Select at least one option "
                   "in each filter.")
        st.stop()

    plot_df = build_plot_df(fdf, view, metric_label, seeds)
    _, lower_better = (RECON_METRICS if view == "recon" else
                       {k: (v[0], v[2]) for k, v in DOWN_METRICS.items()})[metric_label]
    clean_note = (" Dashed line = Clean ceiling."
                  if view == "down" and "Clean" in set(plot_df["method"]) else "")
    st.caption(("Lower is better. " if lower_better else "Higher is better. ")
               + "One panel per mechanism; colour = method. Error bars are ± std "
               + ("across the selected seeds." if view == "recon"
                  else "as stored in the results file (over seeds × folds).")
               + clean_note)

    mechs = ordered(plot_df["mechanism"].unique(), MECH_ORDER)
    panel_methods = [m for m in METHOD_PLOT_ORDER if m in set(plot_df["method"])]
    show_clean = "Clean" in set(plot_df["method"])
    st.markdown(legend_html(panel_methods, show_clean), unsafe_allow_html=True)

    y_max = float(plot_df["upper"].max()) * 1.05
    for i in range(0, len(mechs), 2):
        cols = st.columns(2)
        for col, mech in zip(cols, mechs[i:i + 2]):
            with col:
                st.altair_chart(
                    make_panel(plot_df[plot_df["mechanism"] == mech],
                               metric_label, mech, y_max),
                    use_container_width=True,
                )

    with st.expander("Summary table (plotted values)", expanded=False):
        show = (plot_df[["mechanism", "prop_label", "method", "value", "sd"]]
                .rename(columns={"prop_label": "proportion",
                                 "value": metric_label, "sd": "std"})
                .sort_values(["mechanism", "proportion", "method"]))
        st.dataframe(show, hide_index=True, use_container_width=True)

    with st.expander("Raw filtered rows", expanded=False):
        st.dataframe(fdf, hide_index=True, use_container_width=True)
        st.download_button(
            "Download filtered CSV",
            fdf.to_csv(index=False).encode(),
            file_name=f"{dataset}_{view}_{metric_label.replace(' ', '_')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
