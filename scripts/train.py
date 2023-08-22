#!/usr/bin/python

import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torchcps.gnn import ParametricGNN

from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier
from opf.utils import create_model


def train(params):
    params["data_dir"] = (
        os.path.join(params["root_dir"], "data")
        if params["data_dir"] is None
        else params["data_dir"]
    )
    params["log_dir"] = (
        os.path.join(params["root_dir"], "logs")
        if params["log_dir"] is None
        else params["log_dir"]
    )

    dm = CaseDataModule(pin_memory=params["gpu"], **params)
    model = create_model(params)

    if params["hotstart"]:
        pass

    callbacks = []
    loggers = []
    run_id = os.environ.get("RUN_ID", None)
    if params["log"]:
        if params["wandb"]:
            wandb_logger = WandbLogger(
                project="opf",
                id=run_id,
                save_dir=params["log_dir"],
                config=params,
                log_model=True,
            )
            wandb_logger.log_hyperparams(params)
            wandb_logger.watch(model)
            loggers.append(wandb_logger)
            run_id = wandb_logger.experiment.id

        # logger specific callbacks
        callbacks += [
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=os.path.join(params["log_dir"], f"tensorboard/{run_id}"),
                filename=f"{run_id}"
                + "-epoch={epoch}-loss={val/loss:0.2f}-error={val/inequality/error_max:0.2f}",
                auto_insert_metric_name=False,
            ),
        ]

    callbacks += [EarlyStopping(monitor="val/loss", patience=params["patience"])]
    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator="gpu" if params["gpu"] else "cpu",
        max_epochs=params["max_epochs"],
        default_root_dir=params["log_dir"],
        fast_dev_run=params["fast_dev_run"],
    )

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

    # trainer.tune(model, dm)
    trainer.fit(model, dm)
    trainer.test(model, dm)
    for logger in loggers:
        logger.finalize("finished")


def main():
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument(
        "--hotstart", type=str, default=None, help="ID of run to hotstart with."
    )

    # data arguments
    parser.add_argument("--case_name", type=str, default="case1354_pegase__api")
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
    torch.set_float32_matmul_precision("medium")
    train(vars(params))


if __name__ == "__main__":
    try:
        main()
    finally:
        # exit or the MPS server might be in an undefined state
        torch.cuda.synchronize()
