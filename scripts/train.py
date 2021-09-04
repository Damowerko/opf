#!/usr/bin/python

import os
from pytorch_lightning.loggers import WandbLogger
from opf.utils import model_from_parameters
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--case_name", type=str, default="case30")
parser.add_argument("--adj_threshold", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_epochs", type=int, default=1000)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--K", type=int, default=8)
parser.add_argument("--F", type=int, default=32)
parser.add_argument("--L", type=int, default=2)
parser.add_argument("--s", type=int, default=10)
parser.add_argument("--t", type=int, default=500)
parser.add_argument("--F_MLP", type=int, default=4)
parser.add_argument("--L_MLP", type=int, default=1)
parser.add_argument("--cost_weight", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--eps", type=float, default=1e-4)
parser.add_argument("--constraint-features", type=int, default=False)
parser.add_argument("--root_dir", type=str, default="./")
parser.add_argument("--gpus", type=int, default=1)
params = parser.parse_args()

root_dir = params.root_dir
data_dir = os.path.join(root_dir, "data")
log_dir = os.path.join(root_dir, "logs")


def train(params):
    logger = WandbLogger(project="opf", save_dir=log_dir, config=params, log_model=True)
    barrier, trainer, dm = model_from_parameters(
        params,
        gpus=params["gpus"],
        logger=logger,
        data_dir=data_dir,
        patience=params["patience"],
        eps=params["eps"],
    )

    # figure out the cost weight normalization factor
    calibration_result = trainer.test(barrier, datamodule=dm, verbose=False)[0]
    barrier.cost_normalization = 1.0 / calibration_result["acopf/cost"]

    trainer.tune(barrier, dm)
    trainer.fit(barrier, dm)
    trainer.test(barrier, dm)
    logger.finalize("finished")


if __name__ == "__main__":
    import wandb

    train(vars(params))
    wandb.finish()
