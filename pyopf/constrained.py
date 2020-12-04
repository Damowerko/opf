import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import Dataset, random_split

from GNN.Utils import graphTools
from modules.architectures import SelectionGNN
from pyopf.data import OPFData
from pyopf.modules import OPFLogBarrier

torch.autograd.set_detect_anomaly(True)

os.chdir("..")
case_name = "case30"
data_dir = "data"
ratio_train = 0.9
ratio_valid = 0
device = torch.device("cuda:0")
A_scaling = 0.001
A_threshold = 0.01

data = OPFData(data_dir, case_name, ratio_train, ratio_valid, A_scaling, A_threshold, np.float32, device=device,
               use_constraints=False)
F0 = 4
N = data.case_info()["num_nodes"]
output_size = 4 * N

param = {
    "train": dict(
        max_epochs=20
    ),
    "gnn": dict(
        dimLayersMLP=[output_size],
        dimNodeSignals=[F0, 64],
        nFilterTaps=[4],
        nSelectedNodes=[N],
        poolingSize=[1]
    ),
    "barrier": dict(
        type="relaxed_log",  # log, relaxed_log
        t=200,  # higher -> better approximation to barrier
        s=100,  # the slope at which the barrier becomes linear
        cost_weight=0.01,
    ),
    "meta": dict(
        A_scaling=A_scaling,
        A_threshold=A_threshold,
        model="selection",
        case_name=case_name
    ),
}

adjacencyMatrix = data.getGraph()
G = graphTools.Graph("adjacency", adjacencyMatrix.shape[0], {"adjacencyMatrix": adjacencyMatrix})
G.computeGFT()
S, order = graphTools.permDegree(G.S / np.max(np.real(G.E)))  # normalize GSO by dividing by largest e-value


class Dataset(Dataset):
    def __init__(self, data: OPFData):
        n = data.load.shape[0]
        load = np.zeros((n, data.manager.n_buses, 2))
        load[:, data.manager.load_indices, :] = data.load
        self.load = torch.from_numpy(load)
        self.bus = torch.from_numpy(data.bus)

    def __len__(self): return self.load.shape[0]

    def __getitem__(self, i): return self.bus[i], self.load[i]


def collate_fn(samples):
    buses, loads = zip(*samples)
    return torch.stack(buses).to(torch.float32).cuda(), torch.stack(loads).to(torch.float32).cuda()


gnn = SelectionGNN(GSO=S, **param["gnn"])
barrier = OPFLogBarrier(data.manager, gnn, **param["barrier"]).to("cuda")

dataset = Dataset(data)
split_size = int(len(dataset) * 0.1)
train_data, val_data = random_split(dataset, [len(dataset) - split_size, split_size])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=512, collate_fn=collate_fn)

logger = CometLogger(workspace="damowerko", project_name="opf", save_dir="../logs")
logger.log_hyperparams(param)
trainer = pl.Trainer(logger=logger, gpus=1)
trainer.fit(barrier, train_loader, val_loader)
