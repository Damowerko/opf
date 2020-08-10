import datetime
import itertools
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

import Modules.architectures as archit
import Modules.model as model
import Modules.train as train
import Utils.graphML as gml
import Utils.graphTools as graphTools
from Utils.miscTools import writeVarValues
from opf.data import OPFData
from opf import power
import sklearn as sk

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'serif'
torch.set_default_dtype(torch.float32)

os.chdir("..")

"""
Data
"""
nDataSplits = 1
case_name = "case30"
data_dir = "data"
A_scaling = 0.001
A_threshold = 0.01
ratio_train = 0.8
ratio_valid = 0.1

data = OPFData(data_dir, case_name, ratio_train, ratio_valid, A_scaling, A_threshold, np.float32)

"""
Logging
"""
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
saveDir = os.path.join(data_dir, case_name, "out", today)
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters and results will be saved.
vars_file = os.path.join(saveDir, 'hyperparameters.txt')
with open(vars_file, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))


"""
Training
"""

doSelectionGNN = True

# Individual model training options
trainer = 'ADAM'  # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.005  # In all options
beta1 = 0.9  # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999  # ADAM option only

# Loss function choice
lossFunction = nn.MSELoss()

# Overall training options
nEpochs = 10  # Number of epochs
batchSize = 64  # Batch size
doLearningRateDecay = False  # Learning rate decay
learningRateDecayRate = 0.9  # Rate
learningRateDecayPeriod = 1  # How many epochs after which update the lr
validationInterval = 5  # How many training steps to do the validation

# Save values
writeVarValues(vars_file,
               {'trainer': trainer,
                'learningRate': learningRate,
                'beta1': beta1,
                'beta2': beta2,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

"""
Architectures
"""

# Now that we have accumulated quite an extensive number of architectures, we
# might not want to run all of them. So here we select which ones we want to
# run.

# Select desired node-orderings
doDegree = False
doSpectralProxies = False
doEDS = False
doNoPooling = False

# Select desired architectures
doSelectionGNN = True
doAggregationGNN = True
doMultiNode = False
doGAT = False
doNodeVariantGNN = False
doEdgeVariantGNN = False
doSpectralGNN = False

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. That is, any new architecture in this part, needs also
# to be coded later on. This is just to be easy to change the parameters once
# the architecture is created. Do not forget to add the name of the architecture
# to modelList.


modelList = []

# Parameters to share across several architectures
case_info = data.case_info()

alphas = ([2], [4])
Fs = ([4],[8],[16])
Ks = ([2],[4],[8],[16])
Ms = (2,4)
Ns = ([case_info['num_nodes']],)

parameters = itertools.product(alphas, Fs, Ks, Ms, Ns)
for alpha, F, K, M, N in parameters:
    # MLP layer
    nGenerators = case_info['num_gen']
    dimMLP = [nGenerators]  # MLP after the last layer

    # region MODELS

    # \\\\\\\\\\\\
    # \\\ MODEL 1: Selection GNN ordered by Degree
    # \\\\\\\\\\\\

    if doSelectionGNN:
        hParamsSelGNNDeg = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsSelGNNDeg['name'] = 'SelectionDegree'  # Name of the architecture

        # \\\ Architecture parameters
        hParamsSelGNNDeg['F'] = [data.inputs.shape[1]] + F  # Features per layer
        hParamsSelGNNDeg['K'] = K  # Number of filter taps per layer
        hParamsSelGNNDeg['bias'] = True  # Decide whether to include a bias term
        hParamsSelGNNDeg['sigma'] = torch.nn.ReLU  # Selected nonlinearity
        hParamsSelGNNDeg['N'] = N  # Number of nodes to keep at the end of
        # each layer
        hParamsSelGNNDeg['rho'] = gml.MaxPoolLocal  # Summarizing function
        hParamsSelGNNDeg['alpha'] = alpha  # alpha-hop neighborhood that
        # is affected by the summary
        hParamsSelGNNDeg['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after the GCN layers

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsSelGNNDeg)
        modelList += [hParamsSelGNNDeg['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 2: Aggregation GNN with highest-degree node
    # \\\\\\\\\\\\

    if doAggregationGNN:
        hParamsAggGNNDeg = {}

        hParamsAggGNNDeg['name'] = 'AggregationDegree'

        # \\\ Architecture parameters
        hParamsAggGNNDeg['F'] = [data.inputs.shape[1]] + F  # Features per layer
        hParamsAggGNNDeg['K'] = K  # Number of filter taps per layer
        hParamsAggGNNDeg['bias'] = True  # Decide whether to include a bias term
        hParamsAggGNNDeg['sigma'] = torch.nn.ReLU  # Selected nonlinearity
        hParamsAggGNNDeg['rho'] = torch.nn.MaxPool1d  # Pooling function
        hParamsAggGNNDeg['alpha'] = alpha  # Size of pooling function
        hParamsAggGNNDeg['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after the GCN layers

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsAggGNNDeg)
        modelList += [hParamsAggGNNDeg['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 3: No Pooling Selection GNN
    # \\\\\\\\\\\\

    if doNoPooling:
        ##############
        # PARAMETERS #
        ##############

        hParamsNoPoolGNN = hParamsSelGNNDeg.copy()

        hParamsNoPoolGNN['name'] = 'NoPoolingGNN'

        # \\\ Architecture parameters
        hParamsNoPoolGNN['rho'] = gml.NoPool  # Summarizing function
        hParamsNoPoolGNN['alpha'] = [1, 1]  # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsNoPoolGNN)
        modelList += [hParamsNoPoolGNN['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 4: Spectral GNN
    # \\\\\\\\\\\\

    if doSpectralGNN:
        ##############
        # PARAMETERS #
        ##############

        hParamsSpcGNNDeg = {}  # Hyperparameters (hParams)

        hParamsSpcGNNDeg['name'] = 'SpectralDegree'  # Name of the architecture

        # \\\ Architecture parameters
        hParamsSpcGNNDeg['F'] = [1, F1, F2]  # Features per layer
        hParamsSpcGNNDeg['M'] = [M, M]  # Number of filter taps per layer
        hParamsSpcGNNDeg['bias'] = True  # Decide whether to include a bias term
        hParamsSpcGNNDeg['sigma'] = nn.ReLU  # Selected nonlinearity
        hParamsSpcGNNDeg['N'] = [N1, N2]  # Number of nodes to keep at the end of
        # each layer
        hParamsSpcGNNDeg['rho'] = gml.MaxPoolLocal  # Summarizing function
        hParamsSpcGNNDeg['alpha'] = [alpha1, alpha2]  # alpha-hop neighborhood that
        # is affected by the summary
        hParamsSpcGNNDeg['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after the GCN layers

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsSpcGNNDeg)
        modelList += [hParamsSpcGNNDeg['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 5: Graph Attention Network with No Pooling
    # \\\\\\\\\\\\

    if doGAT:
        hParamsAttNetNoP = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsAttNetNoP['name'] = 'GraphAttention'  # Name of the architecture

        # \\\ Architecture parameters
        hParamsAttNetNoP['F'] = [1, F1, F2]  # Features per layer
        hParamsAttNetNoP['K'] = [K1, K2]  # Number of filter taps per layer
        hParamsAttNetNoP['sigma'] = nn.functional.relu  # Selected nonlinearity
        hParamsAttNetNoP['rho'] = gml.NoPool  # Summarizing function
        hParamsAttNetNoP['alpha'] = [1, 1]  # alpha-hop neighborhood that is
        # affected by the summary
        hParamsAttNetNoP['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after the GCN layers
        hParamsAttNetNoP['bias'] = True  # Decide whether to include a bias term

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsAttNetNoP)
        modelList += [hParamsAttNetNoP['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 6: Node-Variant GNN ordered by Degree, single layer
    # \\\\\\\\\\\\

    if doNodeVariantGNN:
        hParamsNdVGNNDeg = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsNdVGNNDeg['name'] = 'NodeVariantDeg'  # Name of the architecture

        # \\\ Architecture parameters
        hParamsNdVGNNDeg['F'] = [1, F1]  # Features per layer
        hParamsNdVGNNDeg['K'] = [K1]  # Number of shift taps per layer
        hParamsNdVGNNDeg['M'] = [M]  # Number of node taps per layer
        hParamsNdVGNNDeg['bias'] = True  # Decide whether to include a bias term
        hParamsNdVGNNDeg['sigma'] = nn.ReLU  # Selected nonlinearity
        hParamsNdVGNNDeg['rho'] = gml.NoPool  # Summarizing function
        hParamsNdVGNNDeg['alpha'] = [1]  # alpha-hop neighborhood that is
        # affected by the summary
        hParamsNdVGNNDeg['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after the GCN layers

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsNdVGNNDeg)
        modelList += [hParamsNdVGNNDeg['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 7: Full edge-variant
    # \\\\\\\\\\\\

    if doEdgeVariantGNN:
        ##############
        # PARAMETERS #
        ##############

        hParamsEdVGNNNoP = {}

        hParamsEdVGNNNoP['name'] = 'FullEdgeVariant'

        # \\\ Architecture parameters
        hParamsEdVGNNNoP['F'] = [1, F1]  # Features per layer
        hParamsEdVGNNNoP['K'] = [K1]  # Number of shift taps per layer
        hParamsEdVGNNNoP['bias'] = True  # Decide whether to include a bias term
        hParamsEdVGNNNoP['sigma'] = nn.ReLU  # Selected nonlinearity
        hParamsEdVGNNNoP['rho'] = gml.NoPool  # Summarizing function
        hParamsEdVGNNNoP['alpha'] = [1]  # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
        hParamsEdVGNNNoP['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after the GCN layers

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsEdVGNNNoP)
        modelList += [hParamsEdVGNNNoP['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 8: Hybrid edge-variant ordered by Degree
    # \\\\\\\\\\\\

    if doEdgeVariantGNN:
        ##############
        # PARAMETERS #
        ##############

        hParamsEdVGNNDeg = {}

        hParamsEdVGNNDeg['name'] = 'EdgeVariantDegree'

        # \\\ Architecture parameters
        hParamsEdVGNNDeg['F'] = [1, F1]  # Features per layer
        hParamsEdVGNNDeg['K'] = [K1]  # Number of shift taps per layer
        hParamsEdVGNNDeg['M'] = [M]  # Number of selected nodes
        hParamsEdVGNNDeg['bias'] = True  # Decide whether to include a bias term
        hParamsEdVGNNDeg['sigma'] = nn.ReLU  # Selected nonlinearity
        hParamsEdVGNNDeg['rho'] = gml.NoPool  # Summarizing function
        hParamsEdVGNNDeg['alpha'] = [1]  # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
        hParamsEdVGNNDeg['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after the GCN layers

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsEdVGNNDeg)
        modelList += [hParamsEdVGNNDeg['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 9: MultiNode Aggregation GNN by Degree
    # \\\\\\\\\\\\

    if doAggregationGNN and doMultiNode:
        ##############
        # PARAMETERS #
        ##############

        hParamsMNdGNNDeg = {}

        hParamsMNdGNNDeg['name'] = 'MultiNodeDegree'

        # \\\ Architecture parameters
        hParamsMNdGNNDeg['P'] = [4, 2]  # Number of selected nodes
        hParamsMNdGNNDeg['Q'] = [4, 4]  # Number of shifts
        hParamsMNdGNNDeg['F'] = [[1, F1, F1], [F1, F1, F2]]  # Features per layer
        hParamsMNdGNNDeg['K'] = [[K1, K1], [K1, K1]]  # Number of shift taps per layer
        hParamsMNdGNNDeg['bias'] = True  # Decide whether to include a bias term
        hParamsMNdGNNDeg['sigma'] = nn.ReLU  # Selected nonlinearity
        hParamsMNdGNNDeg['rho'] = nn.MaxPool1d  # Pooling function
        hParamsMNdGNNDeg['alpha'] = [[1, 1], [1, 1]]  # Size of pooling function
        hParamsMNdGNNDeg['dimLayersMLP'] = dimMLP.copy()  # Dimension of the fully
        # connected layers after all the aggregation layers

        # \\\ Save Values:
        writeVarValues(vars_file, hParamsMNdGNNDeg)
        modelList += [hParamsMNdGNNDeg['name']]

    # endregion

    """
    Logging
    """

    # Options:
    doPrint = True  # Decide whether to print stuff while running
    doLogging = False  # Log into tensorboard
    doSaveVars = False  # Save (pickle) useful variables
    doFigs = False  # Plot some figures (this only works if doSaveVars is True)
    # Parameters:
    printInterval = 0  # After how many training steps, print the partial results
    #   0 means to never print partial results while training
    xAxisMultiplierTrain = 10  # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
    xAxisMultiplierValid = 2  # How many validation steps in between those shown,
    # same as above.
    figSize = 5  # Overall size of the figure that contains the plot
    lineWidth = 2  # Width of the plot lines
    markerShape = 'o'  # Shape of the markers
    markerSize = 3  # Size of the markers

    writeVarValues(vars_file,
                   {'doPrint': doPrint,
                    'doLogging': doLogging,
                    'doSaveVars': doSaveVars,
                    'doFigs': doFigs,
                    'saveDir': saveDir,
                    'printInterval': printInterval,
                    'figSize': figSize,
                    'lineWidth': lineWidth,
                    'markerShape': markerShape,
                    'markerSize': markerSize})
    """
    Setup
    """

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # Notify:
    if doPrint:
        print("Device selected: %s" % device)

    # \\\ Logging options
    if doLogging:
        # If logging is on, load the tensorboard visualizer and initialize it
        from Utils.visualTools import Visualizer

        logsTB = os.path.join(saveDir, 'logsTB')
        logger = Visualizer(logsTB, name='visualResults')

    # \\\ Save variables during evaluation.
    # We will save all the evaluations obtained for each of the trained models.
    # It basically is a dictionary, containing a list. The key of the
    # dictionary determines the model, then the first list index determines
    # which split realization. Then, this will be converted to numpy to compute
    # mean and standard deviation (across the split dimension).
    accBest = {}  # Accuracy for the best model
    accLast = {}  # Accuracy for the last model
    for thisModel in modelList:  # Create an element for each split realization,
        accBest[thisModel] = [None] * nDataSplits
        accLast[thisModel] = [None] * nDataSplits

    ####################
    # TRAINING OPTIONS #
    ####################

    # Training phase. It has a lot of options that are input through a
    # dictionary of arguments.
    # The value of these options was decided above with the rest of the parameters.
    # This just creates a dictionary necessary to pass to the train function.

    trainingOptions = {}

    if doLogging:
        trainingOptions['logger'] = logger
    if doSaveVars:
        trainingOptions['saveDir'] = saveDir
    if doPrint:
        trainingOptions['printInterval'] = printInterval
    if doLearningRateDecay:
        trainingOptions['learningRateDecayRate'] = learningRateDecayRate
        trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
    trainingOptions['validationInterval'] = validationInterval

    # %%##################################################################
    #                                                                   #
    #                    DATA SPLIT REALIZATION                         #
    #                                                                   #
    #####################################################################

    # Start generating a new data split for each of the number of data splits that
    # we previously specified

    for split in range(nDataSplits):

        # %%##################################################################
        #                                                                   #
        #                    DATA HANDLING                                  #
        #                                                                   #
        #####################################################################

        ############
        # DATASETS #
        ############

        if doPrint:
            print("Loading data", end='')
            if nDataSplits > 1:
                print(" for split %d" % (split + 1), end='')
            print("...", end=' ', flush=True)

        if doPrint:
            print("OK")

        #########
        # GRAPH #
        #########

        if doPrint:
            print("Setting up the graph...", end=' ', flush=True)

        # Create graph
        adjacencyMatrix = data.getGraph()
        G = graphTools.Graph('adjacency', adjacencyMatrix.shape[0],
                             {'adjacencyMatrix': adjacencyMatrix})
        G.computeGFT()  # Compute the GFT of the stored GSO

        # And re-update the number of nodes for changes in the graph (due to
        # enforced connectedness, for instance)
        nNodes = G.N

        # Once data is completely formatted and in appropriate fashion, change its
        # type to torch and move it to the appropriate device
        data.astype(torch.float32)
        data.to(device)

        if doPrint:
            print("OK")

        # %%##################################################################
        #                                                                   #
        #                    MODELS INITIALIZATION                          #
        #                                                                   #
        #####################################################################

        if doPrint:
            print("Creating architectures...", end=' ', flush=True)

        # This is the dictionary where we store the models (in a model.Model
        # class, that is then passed to training).
        modelsGNN = {}

        # Update the models based on the data obtained from the dataset

        if doNoPooling:
            hParamsNoPoolGNN['N'] = [nNodes, nNodes]
        if doGAT:
            hParamsAttNetNoP['N'] = [nNodes, nNodes]
        if doNodeVariantGNN:
            hParamsNdVGNNDeg['N'] = [nNodes]
        if doEdgeVariantGNN:
            hParamsEdVGNNNoP['M'] = [nNodes]
            hParamsEdVGNNNoP['N'] = [nNodes]
            hParamsEdVGNNDeg['N'] = [nNodes]

        # If a new model is to be created, it should be called for here.

        # %%\\\\\\\\\\
        # \\\ MODEL 1: Selection GNN ordered by Degree
        # \\\\\\\\\\\\

        if doSelectionGNN:

            thisName = hParamsSelGNNDeg['name']

            if nDataSplits > 1:
                # Add a new name for this trained model in particular
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permDegree(G.S / np.max(np.real(G.E)))
            # order is an np.array with the ordering of the nodes with respect
            # to the original GSO (the original GSO is kept in G.S).

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.SelectionGNN(  # Graph filtering
                hParamsSelGNNDeg['F'],
                hParamsSelGNNDeg['K'],
                hParamsSelGNNDeg['bias'],
                # Nonlinearity
                hParamsSelGNNDeg['sigma'],
                # Pooling
                hParamsSelGNNDeg['N'],
                hParamsSelGNNDeg['rho'],
                hParamsSelGNNDeg['alpha'],
                # MLP
                hParamsSelGNNDeg['dimLayersMLP'],
                # Structure
                S)
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction  # (if different from default, change
            # it here)

            #########
            # MODEL #
            #########

            SelGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = SelGNNDeg

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 2: Aggregation GNN with highest-degree node
        # \\\\\\\\\\\\

        if doAggregationGNN:

            thisName = hParamsAggGNNDeg['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permDegree(G.S / np.max(np.real(G.E)))

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.AggregationGNN(  # Linear
                hParamsAggGNNDeg['F'],
                hParamsAggGNNDeg['K'],
                hParamsAggGNNDeg['bias'],
                # Nonlinearity
                hParamsAggGNNDeg['sigma'],
                # Pooling
                hParamsAggGNNDeg['rho'],
                hParamsAggGNNDeg['alpha'],
                # MLP in the end
                hParamsAggGNNDeg['dimLayersMLP'],
                # Structure
                S, maxN=10)

            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction

            #########
            # MODEL #
            #########

            AggGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = AggGNNDeg

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 3: No Pooling Selection GNN
        # \\\\\\\\\\\\

        if doNoPooling:

            thisName = hParamsNoPoolGNN['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permIdentity(G.S / np.max(np.real(G.E)))

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.SelectionGNN(  # Graph filtering
                hParamsNoPoolGNN['F'],
                hParamsNoPoolGNN['K'],
                hParamsNoPoolGNN['bias'],
                # Nonlinearity
                hParamsNoPoolGNN['sigma'],
                # Pooling
                hParamsNoPoolGNN['N'],
                hParamsNoPoolGNN['rho'],
                hParamsNoPoolGNN['alpha'],
                # MLP
                hParamsNoPoolGNN['dimLayersMLP'],
                # Structure
                S)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction

            #########
            # MODEL #
            #########

            NoPoolGNN = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = NoPoolGNN

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 4: Spectral GNN
        # \\\\\\\\\\\\

        if doSpectralGNN:

            thisName = hParamsSpcGNNDeg['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permIdentity(G.S / np.max(np.real(G.E)))
            # order is an np.array with the ordering of the nodes with respect
            # to the original GSO (the original GSO is kept in G.S).

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.SpectralGNN(  # Graph filtering
                hParamsSpcGNNDeg['F'],
                hParamsSpcGNNDeg['M'],
                hParamsSpcGNNDeg['bias'],
                # Nonlinearity
                hParamsSpcGNNDeg['sigma'],
                # Pooling
                hParamsSpcGNNDeg['N'],
                hParamsSpcGNNDeg['rho'],
                hParamsSpcGNNDeg['alpha'],
                # MLP
                hParamsSpcGNNDeg['dimLayersMLP'],
                # Structure
                S)
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction  # (if different from default, change
            # it here)

            #########
            # MODEL #
            #########

            SpcGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = SpcGNNDeg

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 5: Graph Attention Network with No Pooling
        # \\\\\\\\\\\\

        if doNoPooling and doGAT:

            thisName = hParamsAttNetNoP['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permIdentity(G.S / np.max(np.real(G.E)))

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GraphAttentionNetwork(  # Graph attentional layers
                hParamsAttNetNoP['F'],
                hParamsAttNetNoP['K'],
                # Nonlinearity
                hParamsAttNetNoP['sigma'],
                # Pooling
                hParamsAttNetNoP['N'],
                hParamsAttNetNoP['rho'],
                hParamsAttNetNoP['alpha'],
                # MLP
                hParamsAttNetNoP['dimLayersMLP'],
                hParamsAttNetNoP['bias'],
                # Structure
                S)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction

            #########
            # MODEL #
            #########

            AttNetNoP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = AttNetNoP

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 6: Node-Variant GNN ordered by Degree
        # \\\\\\\\\\\\

        if doNodeVariantGNN:

            thisName = hParamsNdVGNNDeg['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permDegree(G.S / np.max(np.real(G.E)))
            # order is an np.array with the ordering of the nodes with respect to
            # the original GSO (the original GSO is kept in G.S).

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.NodeVariantGNN(  # Graph filtering
                hParamsNdVGNNDeg['F'],
                hParamsNdVGNNDeg['K'],
                hParamsNdVGNNDeg['M'],
                hParamsNdVGNNDeg['bias'],
                # Nonlinearity
                hParamsNdVGNNDeg['sigma'],
                # Pooling
                hParamsNdVGNNDeg['N'],
                hParamsNdVGNNDeg['rho'],
                hParamsNdVGNNDeg['alpha'],
                # MLP
                hParamsNdVGNNDeg['dimLayersMLP'],
                # Structure
                S)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction

            #########
            # MODEL #
            #########

            NdVGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = NdVGNNDeg

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 7: Full edge-variant
        # \\\\\\\\\\\\

        if doEdgeVariantGNN:

            thisName = hParamsEdVGNNNoP['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permIdentity(G.S / np.max(np.real(G.E)))

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.EdgeVariantGNN(  # Graph filtering
                hParamsEdVGNNNoP['F'],
                hParamsEdVGNNNoP['K'],
                hParamsEdVGNNNoP['M'],
                hParamsEdVGNNNoP['bias'],
                # Nonlinearity
                hParamsEdVGNNNoP['sigma'],
                # Pooling
                hParamsEdVGNNNoP['N'],
                hParamsEdVGNNNoP['rho'],
                hParamsEdVGNNNoP['alpha'],
                # MLP
                hParamsEdVGNNNoP['dimLayersMLP'],
                # Structure
                S)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction

            #########
            # MODEL #
            #########

            EdVGNNNoP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = EdVGNNNoP

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 8: Hybrid edge-variant ordered by Degree
        # \\\\\\\\\\\\

        if doEdgeVariantGNN:

            thisName = hParamsEdVGNNDeg['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permDegree(G.S / np.max(np.real(G.E)))

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.EdgeVariantGNN(  # Graph filtering
                hParamsEdVGNNDeg['F'],
                hParamsEdVGNNDeg['K'],
                hParamsEdVGNNDeg['M'],
                hParamsEdVGNNDeg['bias'],
                # Nonlinearity
                hParamsEdVGNNDeg['sigma'],
                # Pooling
                hParamsEdVGNNDeg['N'],
                hParamsEdVGNNDeg['rho'],
                hParamsEdVGNNDeg['alpha'],
                # MLP
                hParamsEdVGNNDeg['dimLayersMLP'],
                # Structure
                S)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction  # (if different from default, change it here)

            #########
            # MODEL #
            #########

            EdVGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = EdVGNNDeg

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        # %%\\\\\\\\\\
        # \\\ MODEL 9: MultiNode Aggregation GNN by Degree
        # \\\\\\\\\\\\

        if doAggregationGNN and doMultiNode:

            thisName = hParamsMNdGNNDeg['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # \\\ Ordering
            S, order = graphTools.permDegree(G.S / np.max(np.real(G.E)))

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.MultiNodeAggregationGNN(  # Outer structure
                hParamsMNdGNNDeg['P'],
                hParamsMNdGNNDeg['Q'],
                # Graph filtering
                hParamsMNdGNNDeg['F'],
                hParamsMNdGNNDeg['K'],
                hParamsMNdGNNDeg['bias'],
                # Nonlinearity
                hParamsMNdGNNDeg['sigma'],
                # Pooling
                hParamsMNdGNNDeg['rho'],
                hParamsMNdGNNDeg['alpha'],
                # MLP
                hParamsMNdGNNDeg['dimLayersMLP'],
                # Structure
                S)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction  # (if different from default, change it here)

            #########
            # MODEL #
            #########

            MNdGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = MNdGNNDeg

            writeVarValues(vars_file,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        if doPrint:
            print("OK")

        # %%##################################################################
        #                                                                   #
        #                    TRAINING                                       #
        #                                                                   #
        #####################################################################

        ############
        # TRAINING #
        ############

        # On top of the rest of the training options, we pass the identification
        # of this specific data split realization.

        if nDataSplits > 1:
            trainingOptions['graphNo'] = split

        # This is the function that trains the models detailed in the dictionary
        # modelsGNN using the data data, with the specified training options.
        train.MultipleModels(modelsGNN, data,
                             nEpochs=nEpochs, batchSize=batchSize,
                             **trainingOptions)

        # %%##################################################################
        #                                                                   #
        #                    EVALUATION                                     #
        #                                                                   #
        #####################################################################

        # Now that the model has been trained, we evaluate them on the test
        # samples.

        # We have two versions of each model to evaluate: the one obtained
        # at the best result of the validation step, and the last trained model.

        ########
        # DATA #
        ########

        xTest, yTest = data.getSamples('test')

        ##############
        # BEST MODEL #
        ##############

        if doPrint:
            print("Total testing accuracy (Best):", flush=True)

        for key in modelsGNN.keys():
            # Update order and adapt dimensions (this data has one input feature,
            # so we need to add that dimension)

            xTestOrdered = xTest[:, :, modelsGNN[key].order]

            with torch.no_grad():
                # Process the samples
                yHatTest = modelsGNN[key].archit(xTestOrdered)
                # yHatTest is of shape
                #   testSize x numberOfClasses
                # We compute the accuracy
                thisAccBest = data.evaluate(yHatTest, yTest)

            if doPrint:
                print("%s: %6.4f" % (key, thisAccBest), flush=True)

            # Save value
            writeVarValues(vars_file,
                           {'accBest%s' % key: thisAccBest})

            # Now check which is the model being trained
            for thisModel in modelList:
                # If the name in the modelList is contained in the name with
                # the key, then that's the model, and save it
                # For example, if 'SelGNNDeg' is in thisModelList, then the
                # correct key will read something like 'SelGNNDegG01' so
                # that's the one to save.
                if thisModel in key:
                    accBest[thisModel][split] = thisAccBest.item()
                # This is so that we can later compute a total accuracy with
                # the corresponding error.

        ##############
        # LAST MODEL #
        ##############

        # And repeat for the last model

        if doPrint:
            print("Total testing accuracy (Last):", flush=True)

        # Update order and adapt dimensions
        for key in modelsGNN.keys():
            modelsGNN[key].load(label='Last')
            xTestOrdered = xTest[:, :, modelsGNN[key].order]

            with torch.no_grad():
                # Process the samples
                yHatTest = modelsGNN[key].archit(xTestOrdered)
                # yHatTest is of shape
                #   testSize x numberOfClasses
                # We compute the accuracy
                thisAccLast = data.evaluate(yHatTest, yTest)

            if doPrint:
                print("%s: %6.4f" % (key, thisAccLast), flush=True)

            # Save values:
            writeVarValues(vars_file,
                           {'accLast%s' % key: thisAccLast})
            # And repeat for the last model:
            for thisModel in modelList:
                if thisModel in key:
                    accLast[thisModel][split] = thisAccLast.item()

    ############################
    # FINAL EVALUATION RESULTS #
    ############################

    # Now that we have computed the accuracy of all runs, we can obtain a final
    # result (mean and standard deviation)


    meanAccBest = {}  # Mean across data splits
    meanAccLast = {}  # Mean across data splits
    stdDevAccBest = {}  # Standard deviation across data splits
    stdDevAccLast = {}  # Standard deviation across data splits

    if doPrint:
        print("\nFinal evaluations (%02d data splits)" % (nDataSplits))

    for thisModel in modelList:
        # Convert the lists into a nDataSplits vector
        accBest[thisModel] = np.array(accBest[thisModel])
        accLast[thisModel] = np.array(accLast[thisModel])

        # And now compute the statistics (across graphs)
        meanAccBest[thisModel] = np.mean(accBest[thisModel])
        meanAccLast[thisModel] = np.mean(accLast[thisModel])
        stdDevAccBest[thisModel] = np.std(accBest[thisModel])
        stdDevAccLast[thisModel] = np.std(accLast[thisModel])

        # And print it:
        if doPrint:
            print("\t%s: %6.4f (+-%6.4f) [Best] %6.4f (+-%6.4f) [Last]" % (
                thisModel,
                meanAccBest[thisModel],
                stdDevAccBest[thisModel],
                meanAccLast[thisModel],
                stdDevAccLast[thisModel]))

        # Save values
        writeVarValues(vars_file,
                       {'meanAccBest%s' % thisModel: meanAccBest[thisModel],
                        'stdDevAccBest%s' % thisModel: stdDevAccBest[thisModel],
                        'meanAccLast%s' % thisModel: meanAccLast[thisModel],
                        'stdDevAccLast%s' % thisModel: stdDevAccLast[thisModel]})

    # %%##################################################################
    #                                                                   #
    #                    PLOT                                           #
    #                                                                   #
    #####################################################################

    # Finally, we might want to plot several quantities of interest

    if doFigs and doSaveVars:

        ###################
        # DATA PROCESSING #
        ###################

        # Again, we have training and validation metrics (loss and accuracy
        # -evaluation-) for many runs, so we need to carefully load them and compute
        # the relevant statistics from these realizations.

        # \\\ SAVE SPACE:
        # Create the variables to save all the realizations. This is, again, a
        # dictionary, where each key represents a model, and each model is a list
        # for each data split.
        # Each data split, in this case, is not a scalar, but a vector of
        # length the number of training steps (or of validation steps)
        lossTrain = {}
        evalTrain = {}
        lossValid = {}
        evalValid = {}
        # Initialize the splits dimension
        for thisModel in modelList:
            lossTrain[thisModel] = [None] * nDataSplits
            evalTrain[thisModel] = [None] * nDataSplits
            lossValid[thisModel] = [None] * nDataSplits
            evalValid[thisModel] = [None] * nDataSplits

        # \\\ FIGURES DIRECTORY:
        saveDirFigs = os.path.join(saveDir, 'figs')
        # If it doesn't exist, create it.
        if not os.path.exists(saveDirFigs):
            os.makedirs(saveDirFigs)

        # \\\ LOAD DATA:
        # Path where the saved training variables should be
        pathToTrainVars = os.path.join(saveDir, 'trainVars')
        # Get all the training files:
        allTrainFiles = next(os.walk(pathToTrainVars))[2]
        # Go over each of them (this can't be empty since we are also checking for
        # doSaveVars to be true, what guarantees that the variables have been
        # saved.)
        for file in allTrainFiles:
            # Check that it is a pickle file
            if '.pkl' in file:
                # Open the file
                with open(os.path.join(pathToTrainVars, file), 'rb') as fileTrainVars:
                    # Load it
                    thisVarsDict = pickle.load(fileTrainVars)
                    # store them
                    nBatches = thisVarsDict['nBatches']
                    thisLossTrain = thisVarsDict['lossTrain']
                    thisEvalTrain = thisVarsDict['evalTrain']
                    thisLossValid = thisVarsDict['lossValid']
                    thisEvalValid = thisVarsDict['evalValid']
                    # This graph is, actually, the data split dimension
                    if 'graphNo' in thisVarsDict.keys():
                        thisG = thisVarsDict['graphNo']
                    else:
                        thisG = 0
                    # And add them to the corresponding variables
                    for key in thisLossTrain.keys():
                        # This part matches each data realization (matched through
                        # the graphNo key) with each specific model.
                        for thisModel in modelList:
                            if thisModel in key:
                                lossTrain[thisModel][thisG] = thisLossTrain[key]
                                evalTrain[thisModel][thisG] = thisEvalTrain[key]
                                lossValid[thisModel][thisG] = thisLossValid[key]
                                evalValid[thisModel][thisG] = thisEvalValid[key]
        # Now that we have collected all the results, we have that each of the four
        # variables (lossTrain, evalTrain, lossValid, evalValid) has a list for
        # each key in the dictionary. This list goes through the data split.
        # Each split realization is actually an np.array.

        # \\\ COMPUTE STATISTICS:
        # The first thing to do is to transform those into a matrix with all the
        # realizations, so create the variables to save that.
        meanLossTrain = {}
        meanEvalTrain = {}
        meanLossValid = {}
        meanEvalValid = {}
        stdDevLossTrain = {}
        stdDevEvalTrain = {}
        stdDevLossValid = {}
        stdDevEvalValid = {}
        # Initialize the variables
        for thisModel in modelList:
            # Transform into np.array
            lossTrain[thisModel] = np.array(lossTrain[thisModel])
            evalTrain[thisModel] = np.array(evalTrain[thisModel])
            lossValid[thisModel] = np.array(lossValid[thisModel])
            evalValid[thisModel] = np.array(evalValid[thisModel])
            # Each of one of these variables should be of shape
            # nDataSplits x numberOfTrainingSteps
            # And compute the statistics
            meanLossTrain[thisModel] = np.mean(lossTrain[thisModel], axis=0)
            meanEvalTrain[thisModel] = np.mean(evalTrain[thisModel], axis=0)
            meanLossValid[thisModel] = np.mean(lossValid[thisModel], axis=0)
            meanEvalValid[thisModel] = np.mean(evalValid[thisModel], axis=0)
            stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis=0)
            stdDevEvalTrain[thisModel] = np.std(evalTrain[thisModel], axis=0)
            stdDevLossValid[thisModel] = np.std(lossValid[thisModel], axis=0)
            stdDevEvalValid[thisModel] = np.std(evalValid[thisModel], axis=0)

        ####################
        # SAVE FIGURE DATA #
        ####################

        # And finally, we can plot. But before, let's save the variables mean and
        # stdDev so, if we don't like the plot, we can re-open them, and re-plot
        # them, a piacere.
        #   Pickle, first:
        varsPickle = {}
        varsPickle['nEpochs'] = nEpochs
        varsPickle['nBatches'] = nBatches
        varsPickle['meanLossTrain'] = meanLossTrain
        varsPickle['stdDevLossTrain'] = stdDevLossTrain
        varsPickle['meanEvalTrain'] = meanEvalTrain
        varsPickle['stdDevEvalTrain'] = stdDevEvalTrain
        varsPickle['meanLossValid'] = meanLossValid
        varsPickle['stdDevLossValid'] = stdDevLossValid
        varsPickle['meanEvalValid'] = meanEvalValid
        varsPickle['stdDevEvalValid'] = stdDevEvalValid
        with open(os.path.join(saveDirFigs, 'figVars.pkl'), 'wb') as figvars_file:
            pickle.dump(varsPickle, figvars_file)
        #   Matlab, second:
        varsMatlab = {}
        varsMatlab['nEpochs'] = nEpochs
        varsMatlab['nBatches'] = nBatches
        for thisModel in modelList:
            varsMatlab['meanLossTrain' + thisModel] = meanLossTrain[thisModel]
            varsMatlab['stdDevLossTrain' + thisModel] = stdDevLossTrain[thisModel]
            varsMatlab['meanEvalTrain' + thisModel] = meanEvalTrain[thisModel]
            varsMatlab['stdDevEvalTrain' + thisModel] = stdDevEvalTrain[thisModel]
            varsMatlab['meanLossValid' + thisModel] = meanLossValid[thisModel]
            varsMatlab['stdDevLossValid' + thisModel] = stdDevLossValid[thisModel]
            varsMatlab['meanEvalValid' + thisModel] = meanEvalValid[thisModel]
            varsMatlab['stdDevEvalValid' + thisModel] = stdDevEvalValid[thisModel]
        savemat(os.path.join(saveDirFigs, 'figVars.mat'), varsMatlab)

        ########
        # PLOT #
        ########

        # Compute the x-axis
        xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
        xValid = np.arange(0, nEpochs * nBatches, \
                           validationInterval * xAxisMultiplierValid)

        # If we do not want to plot all the elements (to avoid overcrowded plots)
        # we need to recompute the x axis and take those elements corresponding
        # to the training steps we want to plot
        if xAxisMultiplierTrain > 1:
            # Actual selected samples
            selectSamplesTrain = xTrain
            # Go and fetch tem
            for thisModel in modelList:
                meanLossTrain[thisModel] = meanLossTrain[thisModel] \
                    [selectSamplesTrain]
                stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel] \
                    [selectSamplesTrain]
                meanEvalTrain[thisModel] = meanEvalTrain[thisModel] \
                    [selectSamplesTrain]
                stdDevEvalTrain[thisModel] = stdDevEvalTrain[thisModel] \
                    [selectSamplesTrain]
        # And same for the validation, if necessary.
        if xAxisMultiplierValid > 1:
            selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                           xAxisMultiplierValid)
            for thisModel in modelList:
                meanLossValid[thisModel] = meanLossValid[thisModel] \
                    [selectSamplesValid]
                stdDevLossValid[thisModel] = stdDevLossValid[thisModel] \
                    [selectSamplesValid]
                meanEvalValid[thisModel] = meanEvalValid[thisModel] \
                    [selectSamplesValid]
                stdDevEvalValid[thisModel] = stdDevEvalValid[thisModel] \
                    [selectSamplesValid]

        # \\\ LOSS (Training and validation) for EACH MODEL
        for key in meanLossTrain.keys():
            lossFig = plt.figure(figsize=(1.61 * figSize, 1 * figSize))
            plt.errorbar(xTrain, meanLossTrain[key], yerr=stdDevLossTrain[key],
                         color='#01256E', linewidth=lineWidth,
                         marker=markerShape, markersize=markerSize)
            plt.errorbar(xValid, meanLossValid[key], yerr=stdDevLossValid[key],
                         color='#95001A', linewidth=lineWidth,
                         marker=markerShape, markersize=markerSize)
            plt.ylabel(r'Loss')
            plt.xlabel(r'Training steps')
            plt.legend([r'Training', r'Validation'])
            plt.title(r'%s' % key)
            lossFig.savefig(os.path.join(saveDirFigs, 'loss%s.pdf' % key),
                            bbox_inches='tight')

        # \\\ ACCURACY (Training and validation) for EACH MODEL
        for key in meanEvalTrain.keys():
            accFig = plt.figure(figsize=(1.61 * figSize, 1 * figSize))
            plt.errorbar(xTrain, meanEvalTrain[key], yerr=stdDevEvalTrain[key],
                         color='#01256E', linewidth=lineWidth,
                         marker=markerShape, markersize=markerSize)
            plt.errorbar(xValid, meanEvalValid[key], yerr=stdDevEvalValid[key],
                         color='#95001A', linewidth=lineWidth,
                         marker=markerShape, markersize=markerSize)
            plt.ylabel(r'Accuracy')
            plt.xlabel(r'Training steps')
            plt.legend([r'Training', r'Validation'])
            plt.title(r'%s' % key)
            accFig.savefig(os.path.join(saveDirFigs, 'eval%s.pdf' % key),
                           bbox_inches='tight')

        # LOSS (training) for ALL MODELS
        allLossTrain = plt.figure(figsize=(1.61 * figSize, 1 * figSize))
        for key in meanLossTrain.keys():
            plt.errorbar(xTrain, meanLossTrain[key], yerr=stdDevLossTrain[key],
                         linewidth=lineWidth,
                         marker=markerShape, markersize=markerSize)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend(list(meanLossTrain.keys()))
        allLossTrain.savefig(os.path.join(saveDirFigs, 'allLossTrain.pdf'),
                             bbox_inches='tight')

        # ACCURACY (validation) for ALL MODELS
        allEvalValid = plt.figure(figsize=(1.61 * figSize, 1 * figSize))
        for key in meanEvalValid.keys():
            plt.errorbar(xValid, meanEvalValid[key], yerr=stdDevEvalValid[key],
                         linewidth=lineWidth,
                         marker=markerShape, markersize=markerSize)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Training steps')
        plt.legend(list(meanEvalValid.keys()))
        allEvalValid.savefig(os.path.join(saveDirFigs, 'allEvalValid.pdf'),
                             bbox_inches='tight')
