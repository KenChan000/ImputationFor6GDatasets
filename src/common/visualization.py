"""
Visualization + LaTeX table helpers for imputation results.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plot style: match a typical LaTeX thesis (serif body text).
# Applied on import so every figure from this module is consistent.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Canonical ordering used when a caller pins methods explicitly. Kept here so
# Scenario 5 and any other notebook share the same color/order if they opt in.
DEFAULT_METHODS = ["Mean", "kNN", "MICE", "HyperImpute",
                   "SoftImpute", "GRAPE", "DiffPuter"]

cmap = plt.get_cmap("tab10")
METHOD_COLORS = {m: cmap(i) for i, m in enumerate(DEFAULT_METHODS)}


def _as_dataframe(data):
    """Accept a DataFrame or a path-like pointing to a CSV; return a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    raise TypeError(
        f"Expected pandas DataFrame or path to CSV, got {type(data).__name__}"
    )


def _resolve_methods(df: pd.DataFrame, methods) -> list[str]:
    if methods is not None:
        return list(methods)

    present = set(df["method"].unique())
    # Drop baselines that are handled separately by some callers.
    present -= {"Clean", "DropMissing"}

    known_in_order = [m for m in DEFAULT_METHODS if m in present]
    leftover = sorted(present - set(known_in_order))
    return known_in_order + leftover


# ---------------------------------------------------------------------------
# Figure: metric vs proportion as grouped bar chart, one panel per mechanism.
# ---------------------------------------------------------------------------
def plot_metric_bars(data, metric, ylabel, filename=None, methods=None):
    df = _as_dataframe(data)
    methods = _resolve_methods(df, methods)

    scenarios = sorted(df["scenario"].unique())
    proportions = sorted(df["proportion"].unique())

    agg = (df.groupby(["scenario", "proportion", "method"])[metric]
             .agg(["mean", "std"]).reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    # subplots returns a bare Axes (not an array) when there's only one panel.
    if len(scenarios) == 1 and not hasattr(axes, "__iter__"):
        axes = [axes]

    method_color = {m: METHOD_COLORS.get(m, cmap(i)) for i, m in enumerate(methods)}

    n_methods = len(methods)
    n_props = len(proportions)
    bar_width = 0.8 / max(n_methods, 1)
    group_centers = np.arange(n_props)

    for ax, sc in zip(axes, scenarios):
        sub = agg[agg["scenario"] == sc]
        for i, m in enumerate(methods):
            d = (sub[sub["method"] == m]
                 .set_index("proportion")
                 .reindex(proportions))
            offsets = group_centers - 0.4 + (i + 0.5) * bar_width
            ax.bar(offsets, d["mean"], width=bar_width,
                   yerr=d["std"], label=m, color=method_color[m],
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor="black"),
                   alpha=0.9)
        ax.set_title(sc)
        ax.set_xlabel("Missing proportion")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([str(p) for p in proportions])

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(methods), bbox_to_anchor=(0.5, -0.08),
               frameon=False)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Shared cell formatter: "mean ± std" with 3 decimals; drops ± if std is NaN.
# `sep` is the +/- separator: "±" for display tables, "$\\pm$" for LaTeX.
# ---------------------------------------------------------------------------
def _format_cell(mean, std, sep="\u00b1"):
    try:
        if np.isnan(std):
            return f"{mean:.3f}"
    except Exception:
        pass
    return f"{mean:.3f} {sep} {std:.3f}"


# Bold a single cell, in whichever flavour the output wants.
def _bold_latex(text):
    return "\\textbf{" + text + "}"


# CSS used by the Styler to bold winning cells for inline display.
_BOLD_CSS = "font-weight: bold;"


# ---------------------------------------------------------------------------
# Table: RMSE and MAE under a given mechanism, proportions as column groups.
# Each proportion contains two sub-columns: RMSE and MAE.
# ---------------------------------------------------------------------------
def err_table(data, scenario, methods=None, latex=False):
    """RMSE and MAE under one mechanism, proportions as column groups.

    Returns a pandas Styler that renders as a table inline (winning method
    per column bolded). Pass latex=True for the original LaTeX string.
    """
    df = _as_dataframe(data)
    methods = _resolve_methods(df, methods)
    sub = df[df["scenario"] == scenario]

    sep = "$\\pm$" if latex else "\u00b1"

    rows = []
    for metric in ["rmse", "mae"]:
        agg = (sub.groupby(["method", "proportion"])[metric]
                  .agg(["mean", "std"]).reset_index())
        agg["metric"] = metric
        rows.append(agg)
    long = pd.concat(rows)
    long["cell"] = long.apply(
        lambda r: _format_cell(r["mean"], r["std"], sep=sep), axis=1)

    table = long.pivot(index="method",
                       columns=["proportion", "metric"],
                       values="cell")
    table = table.reindex(methods)
    table = table.sort_index(axis=1, level=[0, 1])

    # Record the winner (min mean) per column before renaming columns.
    winners = {}  # (col tuple) -> winning method
    for prop, metric in table.columns:
        means = (sub[sub["proportion"] == prop]
                 .groupby("method")[metric].mean()
                 .reindex(methods))
        winners[(prop, metric)] = means.idxmin()

    pretty_cols = pd.MultiIndex.from_tuples(
        [(f"{int(p*100)}% missing", m.upper()) for p, m in table.columns])

    if latex:
        for col, winner in winners.items():
            table.loc[winner, col] = _bold_latex(table.loc[winner, col])
        table.columns = pd.MultiIndex.from_tuples(
            [(f"{int(p*100)}\\% missing", m.upper()) for p, m in table.columns])
        table.index.name = "Method"
        return table.to_latex(
            escape=False, multicolumn=True, multicolumn_format="c",
            column_format="l" + "cc" * (len(table.columns) // 2),
        )

    pos_winners = [(list(table.columns).index(col), winner)
                   for col, winner in winners.items()]
    table.columns = pretty_cols
    table.index.name = "Method"

    def _highlight(_):
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        for col_pos, winner in pos_winners:
            styles.iloc[table.index.get_loc(winner), col_pos] = _BOLD_CSS
        return styles

    return (table.style
                 .apply(_highlight, axis=None)
                 .set_caption(f"RMSE / MAE under {scenario}"))


# ---------------------------------------------------------------------------
# Table: statistical fidelity at a fixed (mechanism, proportion) slice.
# ---------------------------------------------------------------------------
def stat_fidelity_table(data, scenario="MAR", proportion=0.3,
                        methods=None, latex=False):
    df = _as_dataframe(data)
    methods = _resolve_methods(df, methods)
    sub = df[(df["scenario"] == scenario) & (df["proportion"] == proportion)]

    sep = "$\\pm$" if latex else "\u00b1"

    # pretty label differs only in LaTeX escaping of the dot/backslash.
    metric_specs = [
        ("wasserstein_mean",        "Wasserstein",
         "Wasserstein",      "min"),
        ("mean_shift_mean",         "Mean shift",
         "Mean shift",       "min"),
        ("variance_retention_mean", "Var.\\ retention",
         "Var. retention",   "near_one"),
        ("corr_frob",               "Corr.\\ Frob.",
         "Corr. Frob.",      "min"),
    ]

    columns = {}
    means_by_metric = {}
    rule_lookup = {}
    for raw, latex_name, disp_name, rule in metric_specs:
        name = latex_name if latex else disp_name
        agg = (sub.groupby("method")[raw]
                  .agg(["mean", "std"]).reindex(methods))
        columns[name] = [_format_cell(m, s, sep=sep)
                         for m, s in zip(agg["mean"], agg["std"])]
        means_by_metric[name] = agg["mean"]
        rule_lookup[name] = rule

    table = pd.DataFrame(columns, index=methods)
    table.index.name = "Method"

    winners = {}
    for col in table.columns:
        means = means_by_metric[col]
        if rule_lookup[col] == "min":
            winners[col] = means.idxmin()
        else:  # near_one: closest to 1.0
            winners[col] = (means - 1.0).abs().idxmin()

    if latex:
        for col, winner in winners.items():
            table.loc[winner, col] = _bold_latex(table.loc[winner, col])
        return table.to_latex(escape=False,
                              column_format="l" + "c" * len(table.columns))

    def _highlight(_):
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        for col, winner in winners.items():
            styles.loc[winner, col] = _BOLD_CSS
        return styles

    return (table.style
                 .apply(_highlight, axis=None)
                 .set_caption(
                     f"Statistical fidelity — {scenario}, "
                     f"{int(proportion*100)}% missing"))


# Backwards-compatible alias for the old name.
stat_fidelity_table_latex = stat_fidelity_table