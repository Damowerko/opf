#!/usr/bin/python

import os
from pytorch_lightning.loggers import WandbLogger
from opf.dataset import CaseDataModule
from opf.utils import create_model
import argparse
import pytorch_lightning as pl


def train(params):
    data_dir = os.path.join(params["root_dir"], "data")
    log_dir = os.path.join(params["root_dir"], "logs")

    logger = None
    if not params["no_log"]:
        logger = WandbLogger(
            project="opf", save_dir=log_dir, config=params, log_model=True
        )
        logger.log_hyperparams(params)

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val/loss"),
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=params["patience"]),
    ]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        auto_lr_find=True,
        precision=64,
        gpus=params["gpus"],
        max_epochs=params["max_epochs"],
        gradient_clip_val=params["gradient_clip_val"],
    )

    dm = CaseDataModule(data_dir=data_dir, pin_memory=params["gpus"] != 0, **params)
    model = create_model(dm, params)
    if logger is not None:
        logger.watch(model)

    # figure out the cost weight normalization factor
    # it is chosen so that for any network the IPOPT (ACOPF) cost is 1.0
    calibration_result = trainer.test(model, datamodule=dm, verbose=False)[0]
    model.cost_normalization = 1.0 / calibration_result["acopf/cost"]
    print(f"Cost normalization: {model.cost_normalization}")

    trainer.tune(model, dm)
    trainer.fit(model, dm)
    trainer.test(model, dm)
    if logger is not None:
        logger.finalize("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default="case30")
    parser.add_argument("--adj_threshold", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--s", type=int, default=10)
    parser.add_argument("--t", type=int, default=500)
    parser.add_argument("--cost_weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-4)

    parser.add_argument("--constraint_features", action="store_true")
    parser.add_argument("--enforce_constraints", action="store_true")
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--gradient_clip_val", type=float, default=0)
    parser.add_argument("--hotstart", type=str, default=None)

    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--gpus", type=int, default=1)

    parser.add_argument(
        "--readout", type=str, default="mlp", help="local | mlp | multi"
    )
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--F", type=int, default=32)
    parser.add_argument("--L", type=int, default=2)
    
    params = parser.parse_args()
    print(params)
    train(vars(params))
