import os
from glob import glob

import lightning.pytorch as pl
import torch
import wandb
from tqdm import tqdm

from opf.utils import model_from_parameters

api = wandb.Api()
# runs = api.runs("damowerko/opf", filters={"tag": "foo"})
runs = [
    api.run(f"damowerko/opf/{id}")
    for id in ["2zdzkf7r", "2ng1ftuc", "2lllzf30", "h3lsl9hr"]
]
print(runs)

for run in tqdm(runs):
    param = run.config
    root_dir = "./"
    data_dir = os.path.join(root_dir, "data")
    log_dir = os.path.join(root_dir, "logs")
    logger = pl.loggers.WandbLogger(
        project="opf", save_dir=log_dir, reinit=True, resume="must", id=run.id
    )
    barrier, trainer, dm = model_from_parameters(param, logger=logger)

    files = list(glob(f"{log_dir}/opf/{run.id}/checkpoints/*.ckpt"))
    assert len(files) == 1
    checkpoint = torch.load(files[0], map_location=lambda storage, loc: storage)
    barrier.load_state_dict(checkpoint["state_dict"], strict=False)

    trainer.test(barrier, datamodule=dm, verbose=False)
    logger.finalize("finished")
    wandb.finish()
