import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt
import os
from opf.modules import GNN, OPFLogBarrier
from opf.dataset import CaseDataModule
from multiprocessing import cpu_count
import pytorch_lightning as pl


def graph_info(gso, plot=False):
    print(f"Non-zero edges: {np.sum(np.abs(gso) > 0)}")
    print(f"Connected components: {scipy.sparse.csgraph.connected_components(gso)[0]}")

    if plot:
        plt.figure()
        plt.imshow(gso)
        plt.colorbar()

        plt.figure()
        plt.hist(gso[gso > 0].flat, bins=20, range=(0, 1))
        plt.title("Distribution of non-zero edge weights")
        plt.show()


def model_from_parameters(param, gpus=-1, debug=False, logger=None, data_dir="./data", patience=10, eps=1e-4):
    dm = CaseDataModule(
        param["case_name"],
        data_dir=data_dir,
        batch_size=param["batch_size"],
        num_workers=0 if debug else cpu_count(),
        pin_memory=False,
    )

    input_features = 8 if param["constraint_features"] else 2
    gnn = GNN(
        dm.gso(),
        [input_features] + [param["F"]] * param["gnn_layers"],
        [param["K"]] * param["gnn_layers"],
        [dm.net_wrapper.n_buses * param["MLP"]] * param["mlp_layers"],
    )

    barrier: OPFLogBarrier = OPFLogBarrier(
        dm.net_wrapper,
        gnn,
        t=param["t"],
        s=param["s"],
        cost_weight=param["cost_weight"],
        lr=param["lr"],
        constraint_features=param["constraint_features"],
        eps=eps,
    )

    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val/loss")
    early = pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience)
    trainer = pl.Trainer(
        logger=logger,
        gpus=gpus,
        max_epochs=param["max_epochs"],
        callbacks=[early, model_checkpoint],
        precision=64,
        auto_lr_find=True
    )
    return barrier, trainer, dm
