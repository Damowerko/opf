#!/usr/bin/python

import os
import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, model_checkpoint
from pytorch_lightning import Trainer

from opf.dataset import CaseDataModule
from opf.utils import create_model
from opf.modules import SimpleGNN, OPFLogBarrier


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
    
    dm = CaseDataModule(pin_memory=params["gpus"] != 0, **params)
    model = create_model(dm, params)

    if params["hotstart"]:
        pass

    callbacks = []
    loggers = []
    run_id = os.environ.get("RUN_ID", None)
    if params["log"] and params["wandb"]:
        if params["wandb"]:
            wandb_logger = WandbLogger(
                project="opf",
                id=run_id,
                save_dir=params["log_dir"],
                config=params,
                model_checkpoint=True
            )
            wandb_logger.log_hyperparams(params)
            wandb_logger.watch(model)
            loggers.append(wandb_logger)
            run_id = wandb_logger.experiment.id

        tensorboard_logger = TensorBoardLogger(
            save_dir=params["log_dir"],
            name="tensorboard",
            version=run_id,
            default_hp_metric="test/inequality/error_max"
        )
        loggers.append(tensorboard_logger)

        # logger specific callbacks
        callbacks += [
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=os.path.join(params["log_dir"], f"tensorboard/{run_id}"),
                filename=run_id
                + "-epoch={epoch}-loss={val/loss:0.2f}-error={val/inequality/error_max:0.2f}",
                auto_insert_metric_name=False,
            ),
        ]

    callbacks += [EarlyStopping(monitor="val/loss", patience=params["patience"])]
    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        auto_lr_find=True,
        precision=64,
        gpus=params["gpus"],
        max_epochs=params["max_epochs"],
        default_root_dir=params["log_dir"],
        fast_dev_run=params["fast_dev_run"],
    )

    # figure out the cost weight normalization factor
    # it is chosen so that for any network the IPOPT (ACOPF) cost is 1.0
    if not params["fast_dev_run"]:
        print("Performing cost normalization.")
        calibration_result = trainer.test(model, datamodule=dm, verbose=False)[0]
        model.cost_normalization = 1.0 / calibration_result["acopf/cost"]
        print(f"Cost normalization: {model.cost_normalization}")
    else:
        logging.warning("fast_dev_run is True! Skipping calibration.")

    # trainer.tune(model, dm)
    trainer.fit(model, dm)
    trainer.test(model, dm)
    for logger in loggers:
        logger.finalize("finished")


if __name__ == "__main__":
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
    parser.add_argument("--case_name", type=str, default="case30")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--adj_threshold", type=float, default=0.01)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--patience", type=int, default=50)
    group.add_argument("--gradient_clip_val", type=float, default=0)
    group.add_argument("--gpus", type=int, default=1)

    SimpleGNN.add_args(parser)
    OPFLogBarrier.add_args(parser)

    import sys
    print(sys.argv)
    params = parser.parse_args()
    train(vars(params))
