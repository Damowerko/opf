import h5py
from pathlib import Path
from tqdm.notebook import tqdm
from opf.dataset import CaseDataModule
import opf.powerflow as pf
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
import wandb

import opf
import opf.dataset
import opf.utils
from opf.models import ModelRegistry
from opf.modules import OPFDual
from opf.dataset import PowerflowData


def data_to_device(data: PowerflowData, device: str):
    return PowerflowData(*(x.to(device=device) for x in data))  # type: ignore


def load_run(run_id: str, batch_size: int | None = None):
    run_uri = f"wandb://damowerko-academic/opf/{run_id}"
    base, path = run_uri.split("://")
    if base != "wandb":
        raise ValueError("Only wandb runs are supported")

    api = wandb.Api()
    run = api.run(path)
    config = run.config

    # override some config values
    config.update(data_dir="../data")
    config.update(num_workers=0)
    if batch_size is not None:
        config.update(batch_size=batch_size)
    dm = opf.dataset.CaseDataModule(**config)
    dm.setup()
    assert dm.test_dataset is not None
    assert dm.powerflow_parameters is not None
    n_nodes = (
        dm.powerflow_parameters.n_bus,
        dm.powerflow_parameters.n_branch,
        dm.powerflow_parameters.n_gen,
    )

    model_cls = ModelRegistry.get_class(run.config["model_name"])
    if "combine_branch" not in run.config:
        run.config.update(combine_branch=False)
    model = model_cls(
        metadata=dm.metadata(),
        **run.config,
    )
    opfdual = OPFDual(model, None, n_nodes, n_train=0, **config).cuda()

    # load checkpoint
    with TemporaryDirectory() as tmpdir:
        artifact = api.artifact(f"damowerko-academic/opf/model-{run.id}:best")
        checkpoint_path = artifact.download(root=tmpdir)
        checkpoint = torch.load(
            Path(checkpoint_path) / "model.ckpt", map_location="cpu", weights_only=True
        )
        state_dict = checkpoint["state_dict"]
        state_dict = {
            k.replace("model.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
            if "model." in k
        }
        model.load_state_dict(state_dict, strict=True)
    return dm, opfdual


def test_run(
    run_id: str,
    load_existing: bool = True,
    project=False,
    clamp=False,
    batch_size: int | None = None,
    output_root_path: str = "../data/out",
):
    project_suffix = "_project" if project else ""
    clamp_suffix = "_clamp" if clamp else ""
    data_path = Path(
        f"{output_root_path}/test/{run_id}/test{project_suffix}{clamp_suffix}.parquet"
    )
    if load_existing and data_path.exists():
        return pd.read_parquet(data_path)

    metrics = []
    dm, opfdual = load_run(run_id, batch_size)
    with torch.no_grad(), h5py.File(Path(dm.dataset_path), "r") as f:
        for data in tqdm(dm.test_dataloader()):
            data = data_to_device(data, "cuda")
            graph = data.graph
            variables, _, _ = opfdual(data)
            acopf_cost = (
                f["objective"][data.index.cpu().numpy()]  # type: ignore
                / graph["reference_cost"].cpu().numpy()
            )
            if project:
                variables = opfdual.project_powermodels(variables, graph, clamp=clamp)

            for i in range(graph.num_graphs):
                _graph = graph[i]  # type: ignore
                _variables = pf.PowerflowVariables(
                    variables.V.reshape(graph.num_graphs, -1)[i],
                    variables.S.reshape(graph.num_graphs, -1)[i],
                    variables.Sd.reshape(graph.num_graphs, -1)[i],
                    variables.Sg.reshape(graph.num_graphs, -1)[i],
                    variables.Sg_bus.reshape(graph.num_graphs, -1)[i],
                    variables.Sf.reshape(graph.num_graphs, -1)[i],
                    variables.St.reshape(graph.num_graphs, -1)[i],
                )
                cost = opfdual.cost(_variables, _graph)
                _metrics = {}
                # unnormalized metrics
                constraints = opfdual.constraints(
                    _variables, _graph, None, normalize=False
                )
                for k, v in opfdual.metrics(cost, constraints, "test", True).items():
                    _metrics[k] = v.item()
                # normalized metrics
                constraints_normal = opfdual.constraints(
                    _variables, _graph, None, normalize=True
                )
                for k, v in opfdual.metrics(
                    cost, constraints_normal, "test_normal", True
                ).items():
                    _metrics[k] = v.item()
                _metrics["acopf/cost"] = acopf_cost[i]
                metrics.append(_metrics)
    df = pd.DataFrame(metrics)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path)
    return df
