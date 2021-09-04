import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt
from opf.modules import GNN, OPFLogBarrier
from opf.dataset import CaseDataModule
import pytorch_lightning as pl


def graph_info(gso, plot=False):
    print(f"Non-zero edges: {np.sum(np.abs(gso) > 0)}")
    print(f"Connected components: {scipy.sparse.csgraph.connected_components(gso)[0]}")

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gso)
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.hist(gso[gso > 0].flat, bins=20, range=(0, 1))
        plt.title("Distribution of non-zero edge weights")
        plt.show()


def create_model(dm, params, eps=1e-4):
    input_features = 8 if params["constraint_features"] else 2
    gnn = GNN(
        dm.gso(),
        [input_features] + [params["F"]] * params["L"],
        [params["K"]] * params["L"],
        [dm.net_wrapper.n_buses * params["F_MLP"]] * params["L_MLP"],
    )

    barrier: OPFLogBarrier = OPFLogBarrier(
        dm.net_wrapper,
        gnn,
        t=params["t"],
        s=params["s"],
        cost_weight=params["cost_weight"],
        lr=params["lr"],
        constraint_features=params["constraint_features"],
        eps=eps,
    )
    return barrier


def model_from_parameters(
    params, gpus=-1, debug=False, logger=None, data_dir="./data", patience=10, eps=1e-4
):
    dm = CaseDataModule(
        params["case_name"],
        data_dir=data_dir,
        batch_size=params["batch_size"],
        num_workers=0,  # if debug else cpu_count(),
        pin_memory=gpus,
    )

    input_features = 8 if params["constraint_features"] else 2
    gnn = GNN(
        dm.gso(),
        [input_features] + [params["F"]] * params["L"],
        [params["K"]] * params["L"],
        [dm.net_wrapper.n_buses * params["F_MLP"]] * params["L_MLP"],
    )

    barrier: OPFLogBarrier = OPFLogBarrier(
        dm.net_wrapper,
        gnn,
        t=params["t"],
        s=params["s"],
        cost_weight=params["cost_weight"],
        lr=params["lr"],
        constraint_features=params["constraint_features"],
        eps=eps,
    )

    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val/loss")
    early = pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience)
    trainer = pl.Trainer(
        logger=logger,
        gpus=gpus,
        auto_select_gpus=gpus != 0,
        max_epochs=params["max_epochs"],
        callbacks=[early, model_checkpoint],
        precision=64,
        auto_lr_find=True,
    )
    return barrier, trainer, dm
