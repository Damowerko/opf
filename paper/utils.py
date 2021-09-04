import wandb
from opf.modules import OPFLogBarrier
from opf.utils import model_from_parameters
import torch
import glob
import pytorch_lightning as pl
import pandas as pd

class CacheOutputs(pl.callbacks.Callback):
    def on_test_epoch_start(self, trainer, module):
        self.outputs = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.outputs.append(outputs)

def load_checkpoint(barrier: OPFLogBarrier, id: str, log_dir: str):
    checkpoint_directory = f"{log_dir}opf/{id}/checkpoints/"
    files = list(glob.glob(checkpoint_directory + "*.ckpt"))
    assert len(files) == 1
    checkpoint = torch.load(files[0], map_location=lambda storage, loc: storage)
    barrier.load_state_dict(checkpoint["state_dict"], strict=False)


def test(barrier, dm):
    cache = CacheOutputs()
    trainer = pl.Trainer(precision=64, callbacks=[cache], logger=False)
    trainer.test(barrier, datamodule=dm, verbose=False)
    return pd.DataFrame(cache.outputs).applymap(torch.Tensor.item)