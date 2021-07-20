import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier, GNN
from glob import glob
import wandb
import os
from tqdm import tqdm

api = wandb.Api()
runs = api.runs("damowerko/opf", filters={"tags": "sweep1"})

for run in tqdm(runs):
    param = run.config
    param["root_dir"] = "./"

    data_dir = os.path.join(param["root_dir"], "data")
    log_dir = os.path.join(param["root_dir"], "logs")
    dm = CaseDataModule(
        param["case_name"],
        data_dir=data_dir,
        batch_size=param["batch_size"]
    )
    gnn = GNN(
        dm.gso(),
        [2] + [param["F"]] * param["gnn_layers"],
        [param["K"]] * param["gnn_layers"],
        [dm.net_wrapper.n_buses * param["MLP"]] * param["mlp_layers"]
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
        eps=1e-4
    )

    logger = WandbLogger(
        project="opf",
        save_dir=log_dir,
        reinit=True,
        resume="must",
        id=run.id
    )

    early = pl.callbacks.EarlyStopping(monitor="val/loss", patience=10)
    trainer = pl.Trainer(
        logger=logger,
        gpus=0,
        max_epochs=param["max_epochs"],
        callbacks=[early],
        precision=64,
        auto_lr_find=True
    )

    files = list(glob(f"{log_dir}/opf/{run.id}/checkpoints/*.ckpt"))
    assert len(files) == 1
    checkpoint = torch.load(files[0], map_location=lambda storage, loc: storage)
    barrier.load_state_dict(checkpoint["state_dict"])

    trainer.test(barrier, datamodule=dm, verbose=False)
    logger.finalize("finished")
    wandb.finish()