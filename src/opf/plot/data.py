import pandas as pd
import torch
import wandb
import re

from opf.test import test_run

case_names = [
    "IEEE 30",
    "IEEE 57",
    "IEEE 118",
    "GOC 179",
    # "ACTIV 200",
    "IEEE 300",
]
model_names = [
    "MSE",
    "MSE+Penalty",
    "Dual-S",
    "Dual-S+",
    "Dual-P",
    "Dual-P+",
    "Dual-H",
    "Dual-H+",
]


def load_run_metadata():
    api = wandb.Api()
    runs = api.runs(
        "damowerko-academic/opf",
        filters={
            "$and": [{"config.max_epochs": 5000}, {"created_at": {"$gt": "2025-03-11"}}]
        },
    )

    run_dict = {}
    for run in runs:
        case_name = run.config["case_name"]
        tags = set(run.tags)
        # Rename case names to pretty ones
        if "case30_ieee" in case_name:
            case_name = "IEEE 30"
        elif "case57_ieee" in case_name:
            case_name = "IEEE 57"
        elif "case118_ieee" in case_name:
            case_name = "IEEE 118"
        elif "case179_goc" in case_name:
            case_name = "GOC 179"
        elif "case200_activ" in case_name:
            case_name = "ACTIV 200"
            continue
        elif "case300_ieee" in case_name:
            case_name = "IEEE 300"
        elif "case1354_pegase" in case_name:
            case_name = "PEGASE 1354"
            continue

        best_checkpoint = "best-checkpoint" in tags
        if best_checkpoint:
            tags.remove("best-checkpoint")

        if {"supervised"} == tags:
            model_name = "MSE"
        elif {"supervised", "augmented"} == tags:
            model_name = "MSE+Penalty"
        elif {"pointwise"} == tags:
            model_name = "Dual-P"
        elif {"pointwise", "supervised-warmup"} == tags:
            model_name = "Dual-P+"
        elif {"shared"} == tags:
            model_name = "Dual-S"
        elif {"shared", "supervised-warmup"} == tags:
            model_name = "Dual-S+"
        elif {"hybrid"} == tags:
            model_name = "Dual-H"
        elif {"hybrid", "supervised-warmup"} == tags:
            model_name = "Dual-H+"

        if best_checkpoint and case_name not in [
            # "IEEE 30",
            # "IEEE 57",
            # "IEEE 118",
            "GOC 179",
            # "IEEE 300",
        ]:
            continue

        tags = run.tags
        run_id = run.id
        if case_name not in run_dict:
            run_dict[case_name] = {}
        run_dict[case_name][model_name] = run_id
    return run_dict


def test_or_load_runs(run_dict, output_root_path="../data/out", data_dir="../data/out"):
    torch.set_float32_matmul_precision("high")
    dfs = []
    for case_name in run_dict:
        for model_name, id in run_dict[case_name].items():
            if id == "":
                continue
            # Load test results from consistent /tmp path
            df = test_run(
                id,
                load_existing=True,
                project=True,
                clamp=False,
                output_root_path=output_root_path,
                data_dir=data_dir,
            )
            df = df.assign(id=id, case_name=case_name, model_name=model_name)
            dfs.append(df)

    df = (
        pd.concat(dfs)
        .query("model_name in @model_names")
        .assign(
            model_index=lambda df: df["model_name"].map(lambda x: model_names.index(x)),
            case_index=lambda df: df["case_name"].map(lambda x: case_names.index(x)),
            optimality_gap=lambda df: df["test/cost"] / df["acopf/cost"] - 1,
        )
        .sort_values(["case_index", "model_index"])
        .rename_axis(index="trial")
        .reset_index()
    )
    return df


def select_dual_models(df: pd.DataFrame, dual_model_names: list[str]) -> pd.DataFrame:
    """
    Select dual models from dataframe.
    For each dual type, specify either the `+` or regular model name.
    The function will return only those models which appear in `model_names` with the `+` suffix stripped.

    Args:
        df: pd.DataFrame
        dual_model_names: list of model names to remove
    """
    model_names = ["MSE", "MSE+Penalty"] + dual_model_names
    return df[df["model_name"].isin(model_names)].assign(
        model_name=lambda df: df["model_name"].replace(
            {
                "Dual-H+": "Dual-H",
                "Dual-P+": "Dual-P",
                "Dual-S+": "Dual-S",
            }
        )
    )


def model_summary(df):
    model_names = [model_name for model_name in df["model_name"].unique()]
    index = pd.Index(model_names, name="model_name")
    # First perform the aggregation with simple column names
    df_agg = (
        df.groupby(["model_name"])
        .agg(
            optimality_gap_mean=("optimality_gap", "mean"),
            optimality_gap_std=("optimality_gap", "std"),
            violation_mean_mean=("test_normal/inequality/error_mean", "mean"),
            violation_mean_std=("test_normal/inequality/error_mean", "std"),
            violation_max_mean=("test_normal/inequality/error_max", "mean"),
            violation_max_std=("test_normal/inequality/error_max", "std"),
            violation_max_p95=(
                "test_normal/inequality/error_max",
                lambda x: x.quantile(0.95),
            ),
            violation_max_max=("test_normal/inequality/error_max", "max"),
            # feasible_rate=(
            #     "test_normal/inequality/error_max",
            #     lambda x: (x < 1e-2).mean(),
            # ),
        )
        .reindex(index)
        .rename_axis(index=["Model"])
    )

    # Then create the multiindex columns
    df_agg.columns = pd.MultiIndex.from_tuples(
        [
            ("Optimality Gap %", "Mean"),
            ("Optimality Gap %", "Std."),
            ("Mean Violation %", "Mean"),
            ("Mean Violation %", "Std."),
            ("Max Violation %", "Mean"),
            ("Max Violation %", "Std."),
            ("Max Violation %", "P95"),
            ("Max Violation %", "Max"),
            # ("Feasible Rate %", ""),
        ]
    )
    df_agg = (df_agg * 100).round(2)
    return df_agg


def make_index(df):
    model_names = [model_name for model_name in df["model_name"].unique()]
    index = pd.MultiIndex.from_product(
        [case_names, model_names], names=["Case", "Model"]
    )
    return index


def case_summary(df):
    index = make_index(df)
    # First perform the aggregation with simple column names
    df_agg = (
        df.groupby(["case_name", "model_name"])
        .agg(
            optimality_gap_mean=("optimality_gap", "mean"),
            optimality_gap_std=("optimality_gap", "std"),
            violation_mean_mean=("test_normal/inequality/error_mean", "mean"),
            violation_mean_std=("test_normal/inequality/error_mean", "std"),
            violation_max_mean=("test_normal/inequality/error_max", "mean"),
            violation_max_std=("test_normal/inequality/error_max", "std"),
            violation_max_p95=(
                "test_normal/inequality/error_max",
                lambda x: x.quantile(0.95),
            ),
            violation_max_max=("test_normal/inequality/error_max", "max"),
            # feasible_rate=(
            #     "test_normal/inequality/error_max",
            #     lambda x: (x < 1e-2).mean(),
            # ),
        )
        .reindex(index)
        .rename_axis(index=["Case", "Model"])
    )
    # Then create the multiindex columns
    df_agg.columns = pd.MultiIndex.from_tuples(
        [
            ("Optimality Gap %", "Mean"),
            ("Optimality Gap %", "Std"),
            ("Mean Violation %", "Mean"),
            ("Mean Violation %", "Std"),
            ("Max Violation %", "Mean"),
            ("Max Violation %", "Std"),
            ("Max Violation %", "P95"),
            ("Max Violation %", "Max"),
            # ("Feasible Rate %", ""),
        ]
    )
    df_agg = (df_agg * 100).round(2)
    return df_agg


def constraint_breakdown(df):
    row_index = make_index(df)
    # column index
    constraint_type = {
        "active_power": "Active Generation %",
        "reactive_power": "Reactive Generation %",
        "voltage_magnitude": "Voltage Magnitude %",
        "forward_rate": "Forward Power Flow %",
        "backward_rate": "Backward Power Flow %",
    }
    statistic = ["mean", "max"]
    column_index = pd.MultiIndex.from_product(
        [constraint_type, statistic], names=["Constraint", "Statistic"]
    )

    constraint_pattern = re.compile(
        r"test_normal/(equality|inequality)/([^/]+)/(error_mean|error_max)"
    )
    metric_columns = {}
    for column in df.columns:
        match = constraint_pattern.match(column)
        if not match:
            continue
        _, constraint_name, error_type = match.groups()
        error_type = error_type.replace("error_", "")
        metric_columns[column] = (constraint_name, error_type)

    id_vars = ["case_name", "model_name"]
    df_melted = df[id_vars + list(metric_columns.keys())].melt(
        id_vars=id_vars,
        value_vars=list(metric_columns.keys()),
    )
    df_melted["constraint_type"] = df_melted["variable"].apply(
        lambda x: metric_columns[x][0]
    )
    df_melted["statistic"] = df_melted["variable"].apply(lambda x: metric_columns[x][1])
    df_melted = df_melted.drop(columns=["variable"])

    df_pivoted = (
        df_melted.groupby(id_vars + ["constraint_type", "statistic"])
        .mean()
        .pivot_table(
            index=id_vars,
            columns=["constraint_type", "statistic"],
            values="value",
        )
        .reindex(row_index, columns=column_index)
        .rename(columns=constraint_type)
        .rename(columns={"mean": "Mean", "max": "Max"})
    )
    return df_pivoted * 100
