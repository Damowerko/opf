import pandas as pd
import matplotlib as mpl
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import seaborn.objects as so

from opf.test import test_run
from opf.plot.data import (
    constraint_breakdown,
    load_run_metadata,
    select_dual_models,
    test_or_load_runs,
    case_summary,
    model_summary,
)
from opf.plot.plot import (
    constraint_breakdown_latex,
    plot_tradeoff,
    set_theme_paper,
    so_theme,
    ONE_COLUMN_WIDTH,
    TWO_COLUMN_WIDTH,
    FIGURE_HEIGHT,
    make_summary_table,
)
from pathlib import Path


def main():
    set_theme_paper()

    figure_dir = Path("figures/thesis")
    figure_dir.mkdir(exist_ok=True, parents=True)

    run_dict = load_run_metadata()
    df = test_or_load_runs(run_dict, output_root_path="data/out", data_dir="data/")
    df = select_dual_models(df, ["Dual-S+", "Dual-P+", "Dual-H+"])

    df_case_summary = case_summary(df)
    (figure_dir / "case_summary.tex").write_text(make_summary_table(df_case_summary))

    df_model_summary = model_summary(df)
    (figure_dir / "model_summary.tex").write_text(make_summary_table(df_model_summary))

    df_constraint_breakdown = constraint_breakdown(df)
    (figure_dir / "constraint_breakdown.tex").write_text(
        constraint_breakdown_latex(df_constraint_breakdown)
    )

    p_tradeoff = plot_tradeoff(df, max=True)
    p_tradeoff.save(figure_dir / "tradeoff_max.pdf", bbox_inches="tight")

    p_tradeoff = plot_tradeoff(df, max=False)
    p_tradeoff.save(figure_dir / "tradeoff_mean.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
