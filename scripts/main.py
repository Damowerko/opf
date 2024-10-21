import argparse
import typing
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from wandb.wandb_run import Run

from opf.dataset import CaseDataModule
from opf.hetero import HeteroGCN, OPFReadout
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
        raise NotImplementedError(
            "Removed it. Need to rewrite to avoid confusion between train and optuna."
        )
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
        callbacks += [
            ModelCheckpoint(
                monitor="val/invariant",
                dirpath=Path(params["log_dir"]) / "checkpoints",
                filename="best",
                auto_insert_metric_name=False,
                mode="min",
                save_top_k=1,
            ),
        ]
    callbacks += [EarlyStopping(monitor="val/invariant", patience=params["patience"])]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="cuda" if params["gpu"] else "cpu",
        devices=-1,
        max_epochs=params["max_epochs"],
        default_root_dir=params["log_dir"],
        fast_dev_run=params["fast_dev_run"],
        gradient_clip_val=params["gradient_clip_val"],
    )
    return trainer


def train(trainer: Trainer, params):
    dm = CaseDataModule(pin_memory=params["gpu"], **params)
    if params["homo"]:
        # gcn = GCN(in_channels=dm.feature_dims, out_channels=4, **params)
        raise NotImplementedError("Homogenous model not currently implemented.")
    else:
        dm.setup()
        gcn = HeteroGCN(
            dm.metadata(),
            in_channels=dm.feature_dims,
            out_channels=OPFReadout(dm.metadata(), **params),
            **params,
        )
        if params["compile"]:
            gcn = typing.cast(HeteroGCN, torch.compile(gcn.cuda(), dynamic=True))

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
    # trainer.test(model, dm)
    for logger in trainer.loggers:
        logger.finalize("finished")


if __name__ == "__main__":
    main()
