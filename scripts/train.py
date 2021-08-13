#!/usr/bin/python

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from opf.utils import model_from_parameters

# Default parameters
hyperparameter_defaults = dict(
    case_name="case118",
    adj_scaling="auto",
    adj_threshold=0.01,
    batch_size=256,
    max_epochs=10000,
    patience=100,
    K=8,
    F=32,
    gnn_layers=2,
    MLP=4,
    mlp_layers=1,
    s=10,
    t=500,
    cost_weight=0.1,
    lr=1e-4,
    eps=1e-4,
    constraint_features=False,
)

root_dir="./"
data_dir = os.path.join(root_dir, "data")
log_dir = os.path.join(root_dir, "logs")


def train(param):
    logger = WandbLogger(project="opf", save_dir=log_dir, config=param)
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

    wandb.init(config=hyperparameter_defaults, dir=log_dir, project="opf")
    train(wandb.config)
