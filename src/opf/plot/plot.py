from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seaborn import plotting_context, axes_style
import seaborn.objects as so
import numpy as np

ONE_COLUMN_WIDTH = 3.5
TWO_COLUMN_WIDTH = 7.16
LEGEND_WIDTH = 0.5
FIGURE_HEIGHT = 2.25


def get_rcparams():
    return {
        "figure.figsize": (ONE_COLUMN_WIDTH, FIGURE_HEIGHT),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "lines.linewidth": 0.7,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "legend.framealpha": 1.0,  # Make legend background opaque
        "axes.unicode_minus": False,
    }


def so_theme():
    return (
        plotting_context("paper", font_scale=1.0) | axes_style("ticks") | get_rcparams()
    )


def set_theme_paper():
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.0,
        rc=get_rcparams(),
    )


def add_cmidrule(df: pd.DataFrame, latex_string: str) -> str:
    offset = 1 + df.index.nlevels
    codes = df.columns.codes[0]
    codes = np.pad(codes, (1, 1), mode="constant", constant_values=-1)
    idx = (offset + np.nonzero(codes != np.roll(codes, -1))[0]).tolist()
    cmidrule_str = ""
    for i in range(len(idx) - 1):
        cmidrule_str += rf"\cmidrule(lr){{{idx[i]}-{idx[i+1]-1}}}"
    lines = latex_string.split("\n")
    lines.insert(3, cmidrule_str)
    return "\n".join(lines)


def make_summary_table(df: pd.DataFrame) -> str:
    latex_string = (
        df.style.format(precision=2)
        .format_index(escape="latex", axis=1)
        .hide(axis=0, names=True)
        .to_latex(
            column_format="l" * df.index.nlevels + "c" * len(df.columns),
            multicol_align="c",
            multirow_align="c",
            hrules=True,
            clines="skip-last;data",
        )
    )
    latex_string = add_cmidrule(df, latex_string)
    return latex_string


def constraint_breakdown_latex(df: pd.DataFrame) -> str:
    latex_string = (
        df.rename(
            columns={
                "Active Generation %": r"$\Re(\bfs^g)$ (\%)",
                "Reactive Generation %": r"$\Im(\bfs^g)$ (\%)",
                "Voltage Magnitude %": r"$\lvert V \rvert$ (\%)",
                "Forward Power Flow %": r"$\lvert \bff_{f} \rvert$ (\%)",
                "Backward Power Flow %": r"$\lvert \bff_{t} \rvert$ (\%)",
            }
        )
        .style.format(precision=2)
        .hide(axis=0, names=True)
        .hide(axis=1, names=True)
        .to_latex(
            column_format="l" * df.index.nlevels + "c" * len(df.columns),
            multicol_align="c",
            multirow_align="c",
            hrules=True,
            clines="skip-last;data",
        )
    )
    latex_string = add_cmidrule(df, latex_string)
    return latex_string


def plot_tradeoff(df: pd.DataFrame, max: bool = True):
    if max:
        variable = "test_normal/inequality/error_max"
        variable_label = r"Max Violation (\%)"
    else:
        variable = "test_normal/inequality/error_mean"
        variable_label = r"Mean Violation (\%)"

    # exclude outliers
    pct_min = 0.005
    pct_max = 0.995
    df = (
        df.groupby(["case_name", "model_name"])
        .apply(
            lambda df: df.assign(
                x_pct=df[variable].rank(pct=True),
                y_pct=df["optimality_gap"].rank(pct=True),
            )
        )
        .query(
            f"x_pct > {pct_min} and x_pct < {pct_max} and y_pct > {pct_min} and y_pct < {pct_max}"
        )
    )

    f = plt.figure(figsize=(TWO_COLUMN_WIDTH, 8), layout="tight")
    p = (
        so.Plot(
            data=df.sample(frac=1.0)
            .sort_values("case_index")
            .assign(
                **{
                    variable: lambda df: 100 * df[variable].clip(lower=1e-4),
                    "optimality_gap": lambda df: 100 * df["optimality_gap"],
                }
            ),
            x=variable,
            y="optimality_gap",
            color="model_name",
        )
        .add(so.Dots(pointsize=1.5, alpha=1.0, fillalpha=0.1, stroke=0.2))
        .label(x=variable_label, y=r"Optimality Gap (\%)", color="Model")
        .theme(
            so_theme()
            | {
                "axes.grid.which": "both",
                "legend.markerscale": 5 / 1.5,
            }
        )
        .scale(
            x=so.Continuous(trans="log"),
            y=so.Continuous().label(like=".2f"),
            color=so.Nominal(
                ["tab:pink", "tab:blue", "tab:orange", "tab:red", "tab:green"],
                order=["MSE", "MSE+Penalty", "Dual-S", "Dual-P", "Dual-H"],
            ),
        )
        .share(x=False, y=False)
        .facet(row="case_name")
        .on(f)
        .plot()
    )
    legend = f.legends.pop(0)
    for legend_handle in legend.legend_handles:
        legend_handle.set_linewidth(1.0)

    f.legend(
        legend.legend_handles,
        [t.get_text() for t in legend.texts],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
    )
    plt.close(f)
    return p
