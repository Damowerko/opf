#!/usr/bin/python

import os
from pytorch_lightning.loggers import WandbLogger
from opf.utils import model_from_parameters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--case-name", type=str, default="case30")
parser.add_argument("--adj-threshold", type=float, default=0.01)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--max-epochs", type=int, default=1000)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--K", type=int, default=8)
parser.add_argument("--F", type=int, default=32)
parser.add_argument("--L", type=int, default=2)
parser.add_argument("--s", type=int, default=10)
parser.add_argument("--t", type=int, default=500)
parser.add_argument("--F-MLP", type=int, default=4)
parser.add_argument("--L-MLP", type=int, default=1)
parser.add_argument("--cost-weight", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--eps", type=float, default=1e-4)
parser.add_argument("--constraint-features", type=int, default=False)
parser.add_argument("--root-dir", type=str, default="./")
params = parser.parse_args()

root_dir = params.root_dir
data_dir = os.path.join(root_dir, "data")
log_dir = os.path.join(root_dir, "logs")


def train(param):
    logger = WandbLogger(project="opf", save_dir=log_dir, config=param, log_model=True)
    barrier, trainer, dm = model_from_parameters(
        param,
        gpus=1,
        logger=logger,
        data_dir=data_dir,
        patience=param["patience"],
        eps=param["eps"],
    )
    logger.watch(barrier)

    trainer.tune(barrier, datamodule=dm)
    trainer.fit(barrier, dm)
    trainer.test(datamodule=dm)
    logger.finalize("finished")
    wandb.finish()


if __name__ == "__main__":
    import wandb

    wandb.init(config=params, dir=log_dir, project="opf")
    train(wandb.config)
