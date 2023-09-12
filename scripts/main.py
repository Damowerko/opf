import argparse
import os
import typing
from functools import partial
from pathlib import Path

import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torchcps.gnn import ParametricGNN

from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier
from opf.utils import create_model


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

    # data arguments
    parser.add_argument("--case_name", type=str, default="case179_goc__api")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--gpu", type=bool, default=True)
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--patience", type=int, default=50)
    group.add_argument("--gradient_clip_val", type=float, default=0)

    # the gnn being used
    ParametricGNN.add_args(parser)
    OPFLogBarrier.add_args(parser)

    params = parser.parse_args()
    params_dict = vars(params)

    # parse the data dir

    torch.set_float32_matmul_precision("medium")
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
            **wandb_kwargs,
        )
        logger.log_hyperparams(params)

        # logger specific callbacks
        callbacks += [
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=Path(params["log_dir"]) / "checkpoints",
                filename="best",
                auto_insert_metric_name=False,
                mode="min",
                save_top_k=1,
            ),
        ]
    callbacks += [EarlyStopping(monitor="val/loss", patience=params["patience"])]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu" if params["gpu"] else "cpu",
        devices=1,
        max_epochs=params["max_epochs"],
        default_root_dir=params["log_dir"],
        fast_dev_run=params["fast_dev_run"],
    )
    return trainer


def train(trainer: Trainer, params):
    dm = CaseDataModule(pin_memory=params["gpu"], **params)
    model = create_model(params)

    # TODO: Add back once we can run ACOPF examples.
    # figure out the cost weight normalization factor
    # it is chosen so that for any network the IPOPT (ACOPF) cost is 1.0
    # if not params["fast_dev_run"]:
    #     print("Performing cost normalization.")
    #     calibration_result = trainer.test(model, datamodule=dm, verbose=False)[0]
    #     model.cost_normalization = 1.0 / calibration_result["acopf/cost"]
    #     print(f"Cost normalization: {model.cost_normalization}")
    # else:
    #     logging.warning("fast_dev_run is True! Skipping calibration.")

    # find the 'optimal' learning rate
    # tuner = Tuner(trainer)
    # tuner.lr_find(model, dm)

    trainer.fit(model, dm)
    # trainer.test(model, dm)
    for logger in trainer.loggers:
        logger.finalize("finished")


def study(params: dict):
    torch.set_float32_matmul_precision("medium")
    study_name = "opf-3"
    storage = os.environ["OPTUNA_STORAGE"]
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10, max_resource="auto", reduction_factor=3
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
        directions=["minimize"],
    )
    study.optimize(
        partial(objective, default_params=params),
        n_trials=1,
    )


def objective(trial: optuna.trial.Trial, default_params: dict):
    params: dict[str, typing.Any] = dict(
        lr=trial.suggest_float("lr", 1e-8, 1e-1, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-16, 1, log=True),
        n_layers=trial.suggest_int("n_layers", 1, 128),
        dropout=trial.suggest_float("dropout", 0, 1),
        heads=8,
        n_channels=32,
        activation="leaky_relu",
        # loss function parameters, should be kept constant within study
        cost_weight=1e-6,
        equality_weight=1e3,
        # MLP parameteers
        mlp_hidden_channels=256,
        mlp_read_layers=trial.suggest_int("mlp_read_layers", 1, 4),
        mlp_per_gnn_layers=trial.suggest_int("mlp_per_gnn_layers", 0, 4),
        enforce_constraints=True,
    )
    params = {**default_params, **params}
    trainer = make_trainer(
        params,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val/loss"
            )
        ],
        wandb_kwargs=dict(group=trial.study.study_name),
    )

    dm = CaseDataModule(pin_memory=params["gpu"], **params)
    model = create_model(params)
    trainer.fit(model, dm)

    # finish up
    if isinstance(trainer.logger, WandbLogger):
        trial.set_user_attr("wandb_id", trainer.logger.experiment.id)

    for logger in trainer.loggers:
        logger.finalize("finished")
    print(trainer.callback_metrics)
    return trainer.callback_metrics["val/loss"].item()


if __name__ == "__main__":
    try:
        main()
    finally:
        # exit or the MPS server might be in an undefined state
        torch.cuda.synchronize()
