#!/usr/bin/python

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier, GNN
from multiprocessing import cpu_count

# Default parameters
hyperparameter_defaults = dict(
    case_name="case30",
    adj_scaling="auto",
    adj_threshold=0.01,
    batch_size=1024,
    max_epochs=100,
    K=4,
    F=16,
    gnn_layers=1,
    MLP=4,
    mlp_layers=1,
    t=20,
    s=10,
    cost_weight=1.0,
    lr=3e-4,
    constraint_features=False,
    root_dir="../"
)

data_dir = os.path.join(hyperparameter_defaults["root_dir"], "data")
log_dir = os.path.join(hyperparameter_defaults["root_dir"], "logs")


def train(param):
    dm = CaseDataModule(
        param["case_name"],
        data_dir=data_dir,
        batch_size=param["batch_size"],
        ratio_train=0.8,
        num_workers=cpu_count(),
        pin_memory=True
    )

    gnn = GNN(
        dm.gso(),
        [2] + [param["F"]] * param["gnn_layers"],
        [param["K"]] * param["gnn_layers"],
        [dm.net_wrapper.n_buses * param["MLP"]] * param["mlp_layers"]
    ).float()

    barrier = OPFLogBarrier(
        dm.net_wrapper,
        gnn,
        t=param["t"],
        s=param["s"],
        cost_weight=param["cost_weight"],
        type="relaxed_log",
        lr=param["lr"],
        constraint_features=param["constraint_features"]
    ).float()

    logger = WandbLogger(project="opf", save_dir=log_dir)
    logger.watch(barrier)

    trainer = pl.Trainer(logger=logger,
                         gpus=-1,
                         max_epochs=param["max_epochs"])
    trainer.fit(barrier, dm)
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    import wandb

    wandb.init(config=hyperparameter_defaults, dir=log_dir, project="opf")
    train(wandb.config)
