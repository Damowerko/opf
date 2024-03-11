import time

import numpy as np
import torch
from scipy.stats import zscore
from torch_geometric.transforms import ToSparseTensor
from torchcps.gnn import GCN
from tqdm import trange

from opf.dataset import CaseDataModule, StaticGraphDataset

# load IPOPT runtime data
ipopt_times = np.load("data/out/times_ipopt_case118_ieee.npz")

# For each power system, we trained a GNN with L = 2 layers,
# K = 8th order filters, F = 32 features at each hidden layer

gnn = GCN(
    2,
    4,
    n_taps=8,
    n_layers=2,
    n_channels=32,
    mlp_read_layers=1,
    mlp_hidden_channels=32,
).cuda()

# load data sample
dm = CaseDataModule(case_name="case118_ieee", num_workers=0, pin_memory=True, homo=True)
dm.setup("test")
dataset = dm.test_dataset
assert isinstance(dataset, StaticGraphDataset)
data = dataset[0].data.cuda()
data.edge_weight = data.edge_attr[:, 0]
data = ToSparseTensor()(data)

n_samples = len(ipopt_times)
gnn_times = np.zeros(n_samples)

gnn.eval()
for i in trange(len(gnn_times)):
    start = time.time()
    with torch.no_grad():
        y = gnn(data.x[:, :2], data.adj_t)
    torch.cuda.synchronize()
    gnn_times[i] = time.time() - start

# remove outliers
gnn_times = gnn_times[np.abs(zscore(gnn_times)) < 3.0] * 1000
ipopt_times = ipopt_times[np.abs(zscore(ipopt_times)) < 3.0] * 1000

# mean and std for gnn and ipopt
print(f"GNN  : {np.mean(gnn_times):8.4f} ± {np.std(gnn_times):.4f} ms")
print(f"IPOPT: {np.mean(ipopt_times):8.4f} ± {np.std(ipopt_times):.4f} ms")

# number of parameters in GNN
total_params = sum([np.prod(p.size()) for p in gnn.parameters()])
print(f"GNN Parameters: {total_params}")
