import sklearn
import skorch
from opf.data import OPFData
from GNN.Modules.architectures import SelectionGNN
from GNN.Utils import graphTools
from GNN.Utils import graphML
import numpy as np
import torch
import os
import pandas as pd
from time import gmtime, strftime
from opf.modules import LocalGNN

os.chdir("..")

case_name = "case30"
data_dir = "data"
ratio_train = 0.9
ratio_valid = 0
device = 'cuda:0'

param_meta = {
    'A_scaling': 0.001,
    'A_threshold': 0.01,
    'local': False
}

data = OPFData(data_dir, case_name, ratio_train, ratio_valid, param_meta['A_scaling'], param_meta['A_threshold'],
               np.float32, device=device)
case_info = data.case_info()
N = case_info['num_nodes']

param_grid = {
    'module__dimNodeSignals': [[4, 64, 32], [4, 128, 32], [4, 128, 64], [4, 128, 128]],
    'module__nFilterTaps': [[2, 2], [4, 4], [6, 6], [8, 8]],
    'module__poolingSize': [[1, 1]],
    'module__nSelectedNodes': [[N, N]],
    'module__dimLayersMLP': [[]],
    'max_epochs': [50],
}

#param_fit = {'max_epochs': 200, 'module__dimNodeSignals': [4], 'module__nFilterTaps': [], 'module__dimLayersMLP': [512, 512, case_info['num_gen']], 'module__nSelectedNodes': [], 'module__poolingSize': []}
param_fit = {'max_epochs': 200, 'module__dimLayersMLP': [5], 'module__dimNodeSignals': [4, 64, 32], 'module__nFilterTaps': [6, 6], 'module__nSelectedNodes': [30, 30], 'module__poolingSize': [1, 1]}
#param_fit = {}

adjacencyMatrix = data.getGraph()
G = graphTools.Graph('adjacency', adjacencyMatrix.shape[0], {'adjacencyMatrix': adjacencyMatrix})
G.computeGFT()
S, order = graphTools.permDegree(G.S / np.max(np.real(G.E)))  # normalize GSO by dividing by largest e-value


def cost(model, x, y):
    yHat = model.predict(x)
    return data.cost_percentage(yHat, y)


def rms(model, x, y):
    yHat = model.predict(x)
    return data.rms(yHat, y)


power_callback = skorch.callbacks.EpochScoring(cost)
rms_callback = skorch.callbacks.EpochScoring(rms)

if param_meta['local']:
    del param_grid['module__nSelectedNodes']
    del param_grid['module__poolingSize']
    del param_grid['module__dimLayersMLP']
    param_grid['module__dimNodeSignals'] = [F + [1] for F in param_grid['module__dimNodeSignals']]
    param_grid['module__nFilterTaps'] = [K + [1] for K in param_grid['module__nFilterTaps']]
    net = skorch.NeuralNetRegressor(LocalGNN, optimizer=torch.optim.Adam, device=device, callbacks=[rms_callback],
                                    module__GSO=S, module__index=case_info['gen_index'], **param_fit)
else:
    param_grid['module__dimLayersMLP'] = [M + [case_info['num_gen']] for M in param_grid['module__dimLayersMLP']]
    net = skorch.NeuralNetRegressor(SelectionGNN, optimizer=torch.optim.Adam, module__GSO=S, device=device,
                                    callbacks=[rms_callback], **param_fit)

x_train, y_train = data.getSamples('train')
x_test, y_test = data.getSamples('test')
x_train = x_train[:, :, order]
x_test = x_test[:, :, order]

if len(param_fit) == 0:
    rms_scorer = sklearn.metrics.make_scorer(lambda y, y_pred: data.rms(y, y_pred), False)
    gs = sklearn.model_selection.GridSearchCV(net, param_grid, refit=False, cv=3, scoring=rms_scorer)
    gs.fit(x_train, y_train)
    print(gs.best_score_, gs.best_params_)
    scores_df = pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score')
    for key in param_meta.keys():
        scores_df[key] = param_meta[key]
    time = strftime("%y-%m-%d-%H%M", gmtime())
    scores_df.to_excel(os.path.join(data_dir, case_name, time + ".xlsx"))
else:
    net.fit(x_train, y_train)
    cost, violated_rate = cost(net, x_train, y_train)
    print("RMS: {} | Cost ratio {} | Violation Rate {}".format(rms(net, x_train, y_train), cost, violated_rate))
