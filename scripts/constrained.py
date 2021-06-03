#!/usr/bin/python

# noinspection PyUnresolvedReferences
import comet_ml
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier, GNN
from opf.utils import graph_info

if __name__ == "__main__":
    # %%

    # constants
    data_dir = "../data"

    # parameters that should be saved for logging
    param = dict(
        case_name="case30",
        A_scaling=300,
        A_threshold=0.01,
        model="selection",
        batch_size=128,
        num_workers=8,
        pin_memory=True,
    )
    dm = CaseDataModule(
        param["case_name"],
        data_dir=data_dir,
        batch_size=param["batch_size"],
        ratio_train=0.8,
    )

    # %% Load Dataset

    # Choose scaling factor so that the mean weight is 0.5
    param["A_scaling"] = (
        2 * np.exp(-1) / np.mean(dm.net_wrapper.impedence_matrix().data)
    )
    adjacency = dm.adjacency(param["A_scaling"], param["A_threshold"])
    graph_info(adjacency, plot=True)
    # Normalize GSO by dividing by larget eigenvalue
    gso = adjacency / np.max(np.real(np.linalg.eigh(adjacency)[0]))

    adjacency = dm.adjacency(param["A_scaling"], param["A_threshold"])
    graph_info(adjacency)
    # Normalize GSO by dividing by larget eigenvalue
    gso = adjacency / np.max(np.real(np.linalg.eigh(adjacency)[0]))

    # %%

    gnn = GNN(gso, [2, 64], [4], [4 * dm.size(1)]).float()

    barrier = OPFLogBarrier(
        dm.net_wrapper,
        gnn,
        t=200,
        s=100,
        cost_weight=1.0,
        type="relaxed_log",
    ).float()

    # %%

    logger = CometLogger(
        workspace="damowerko",
        project_name="opf",
        save_dir="../logs",
        display_summary_level=0,
    )
    logger.log_hyperparams(param)
    logger.experiment.log_code(folder="../src")
    logger.experiment.log_code(folder="../scripts")

    trainer = pl.Trainer(
        logger=logger,
        progress_bar_refresh_rate=1,
        gpus=1,
        max_epochs=100,
        auto_lr_find=True,
    )

    trainer.tune(barrier, dm)
    trainer.fit(barrier, dm)
    trainer.test(datamodule=dm)
    trainer.save_checkpoint("latest.ckpt")
