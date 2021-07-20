#!/usr/bin/python

import os
import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.loggers import WandbLogger
from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier, GNN
from multiprocessing import cpu_count

# Default parameters
hyperparameter_defaults = dict(
    case_name="case30",
    adj_scaling="auto",
    adj_threshold=0.01,
    batch_size=2048,
    max_epochs=1000,
    K=8,
    F=32,
    gnn_layers=2,
    MLP=4,
    mlp_layers=1,
    t=10,
    s=500,
    cost_weight=0.1,
    lr=0.0001,
    constraint_features=False,
    root_dir="./",
)

data_dir = os.path.join(hyperparameter_defaults["root_dir"], "data")
log_dir = os.path.join(hyperparameter_defaults["root_dir"], "logs")


def train(param):

    data_dir = os.path.join(param["root_dir"], "data")
    log_dir = os.path.join(param["root_dir"], "logs")

    dm = CaseDataModule(
        param["case_name"],
        data_dir=data_dir,
        batch_size=param["batch_size"],
        num_workers=cpu_count(),
        pin_memory=False,
    )

    gnn = GNN(
        dm.gso(),
        [2] + [param["F"]] * param["gnn_layers"],
        [param["K"]] * param["gnn_layers"],
        [dm.net_wrapper.n_buses * param["MLP"]] * param["mlp_layers"],
    )

    # noinspection PyTypeChecker
    barrier: OPFLogBarrier = OPFLogBarrier(
        dm.net_wrapper,
        gnn,
        t=param["t"],
        s=param["s"],
        cost_weight=param["cost_weight"],
        lr=param["lr"],
        constraint_features=param["constraint_features"],
        eps=1e-4,
    )

    logger = WandbLogger(project="opf", save_dir=log_dir)
    logger.watch(barrier)

    early = pl.callbacks.EarlyStopping(monitor="val/loss", patience=10)
    trainer = pl.Trainer(
        logger=logger,
        gpus=1,
        max_epochs=param["max_epochs"],
        callbacks=[early],
        precision=64,
        auto_lr_find=True
    )

    trainer.fit(barrier, dm)
    trainer.test(datamodule=dm)
    logger.finalize("finished")
    wandb.finish()

if __name__ == "__main__":
    import wandb

    wandb.init(config=hyperparameter_defaults, dir=log_dir, project="opf")
    train(wandb.config)
