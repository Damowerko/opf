import argparse
import os
import typing
from functools import partial
from pathlib import Path

import optuna
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb.wandb_run import Run

from opf.dataset import CaseDataModule
from opf.hetero import HeteroGCN
from opf.modules import OPFDual


def main():
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("operation", choices=["train", "study"])
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--no_log", action="store_false", dest="log")
    parser.add_argument(
        "--hotstart", type=str, default=None, help="ID of run to hotstart with."
    )
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--no_compile", action="store_false", dest="compile", default=True
    )
    parser.add_argument(
        "--no_personalize", action="store_false", dest="personalize", default=True
    )

    # data arguments
    parser.add_argument("--case_name", type=str, default="case179_goc__api")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--homo", action="store_true", default=False)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--no_gpu", action="store_false", dest="gpu")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--patience", type=int, default=50)
    group.add_argument("--gradient_clip_val", type=float, default=0)

    # the gnn being used
    HeteroGCN.add_args(parser)
    OPFDual.add_args(parser)

    params = parser.parse_args()
    params_dict = vars(params)

    torch.set_float32_matmul_precision("high")
    if params.operation == "study":
        study(params_dict)
    elif params.operation == "train":
        train(make_trainer(params_dict), params_dict)
    else:
        raise ValueError(f"Unknown operation: {params.operation}")


def make_trainer(params, callbacks=[], wandb_kwargs={}):
    logger = None
    if params["log"]:
        logger = WandbLogger(
            project="opf",
            save_dir=params["log_dir"],
            config=params,
            log_model=True,
            notes=params["notes"],
            **wandb_kwargs,
        )

        logger.log_hyperparams(params)
        typing.cast(Run, logger.experiment).log_code(
            Path(__file__).parent.parent,
            include_fn=lambda path: (
                (path.endswith(".py") or path.endswith(".jl"))
                and "logs" not in path
                and ("src" in path or "scripts" in path)
            ),
        )

        # logger specific callbacks
        # callbacks += [
        #     ModelCheckpoint(
        #         monitor="val/invariant",
        #         dirpath=Path(params["log_dir"]) / "checkpoints",
        #         filename="best",
        #         auto_insert_metric_name=False,
        #         mode="min",
        #         save_top_k=1,
        #     ),
        # ]
    callbacks += [EarlyStopping(monitor="val/invariant", patience=params["patience"])]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=32,
        accelerator="cuda" if params["gpu"] else "cpu",
        devices=1,
        max_epochs=params["max_epochs"],
        default_root_dir=params["log_dir"],
        fast_dev_run=params["fast_dev_run"],
        gradient_clip_val=params["gradient_clip_val"],
        log_every_n_steps=1,
    )
    return trainer


def _train(trainer: Trainer, params):
    dm = CaseDataModule(pin_memory=params["gpu"], **params)
    if params["homo"]:
        # gcn = GCN(in_channels=dm.feature_dims, out_channels=4, **params)
        raise NotImplementedError("Homogenous model not currently implemented.")
    else:
        dm.setup()
        gcn = HeteroGCN(
            dm.metadata(),
            in_channels=max(dm.feature_dims.values()),
            out_channels=4,
            **params,
        )
        if params["compile"]:
            gcn = typing.cast(HeteroGCN, torch.compile(gcn.cuda()))

    assert dm.powerflow_parameters is not None
    n_nodes = (
        dm.powerflow_parameters.n_bus,
        dm.powerflow_parameters.n_branch,
        dm.powerflow_parameters.n_gen,
    )
    model = OPFDual(
        gcn, n_nodes, multiplier_table_length=len(dm.train_dataset) if params["personalize"] else 0, **params  # type: ignore
    )
    trainer.fit(model, dm)


def train(trainer: Trainer, params):
    _train(trainer, params)
    # trainer.test(model, dm)
    for logger in trainer.loggers:
        logger.finalize("finished")


def study(params: dict):
    study_name = "opf-4"
    storage = os.environ["OPTUNA_STORAGE"]
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=50, max_resource=1000, reduction_factor=3
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True,
        directions=["minimize"],
    )
    study.optimize(
        partial(objective, default_params=params),
        n_trials=1,
    )


def objective(trial: optuna.trial.Trial, default_params: dict):
    params = dict(
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-16, 1, log=True),
        lr_dual=trial.suggest_float("lr_dual", 1e-5, 1.0, log=True),
        lr_common=trial.suggest_float("lr_common", 1e-5, 1.0, log=True),
        weight_decay_dual=trial.suggest_float("weight_decay_dual", 1e-16, 1, log=True),
        dropout=trial.suggest_float("dropout", 0, 1),
        augmented_weight=10.0,
        supervised_weight=100.0,
        powerflow_weight=1.0,
        case_name="case118_ieee",
        n_layers=20,
        batch_size=100,
        n_channels=128,
        cost_weight=1.0,
        equality_weight=1e3,
        max_epochs=1000,
        patience=1000,
        warmup=10,
        supervised_warmup=20,
        # # MLP parameteers
        mlp_hidden_channels=512,
        mlp_read_layers=2,
        mlp_per_gnn_layers=2,
    )
    params = {**default_params, **params}
    trainer = make_trainer(
        params,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val/invariant"
            )
        ],
        wandb_kwargs=dict(group=trial.study.study_name),
    )

    train(trainer, params)

    # finish up
    if isinstance(trainer.logger, WandbLogger):
        trial.set_user_attr("wandb_id", trainer.logger.experiment.id)
    for logger in trainer.loggers:
        logger.finalize("finished")

    print(trainer.callback_metrics)
    return trainer.callback_metrics["val/invariant"].item()


if __name__ == "__main__":
    main()
