from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import itertools
from cnn import CNN
import random
from utils import *

def trainDevSplit(reviews, x, y, trainPercentage=0.70, devPercentage=0.15):
    # Get dimensions
    numClasses = len(np.unique(y))
    
    # Stratified split keeps class distribution
    if isinstance(x, np.ndarray):
        for trainIndices, devIndices in StratifiedShuffleSplit(y, n_iter=1, test_size=devPercentage):
            reviewsTrain, xTrain, yTrain = reviews[trainIndices], x[trainIndices], y[trainIndices]
            reviewsDev, xDev, yDev = reviews[devIndices], x[devIndices], y[devIndices]
    else:
        for trainIndices, devIndices in StratifiedShuffleSplit(y, n_iter=1, test_size=devPercentage):
            reviewsTrain = [reviews[i] for i in trainIndices]
            xTrain = [x[i] for i in trainIndices]
            yTrain = [y[i] for i in trainIndices]
            reviewsDev =[reviews[i] for i in devIndices]
            xDev = [x[i] for i in devIndices]
            yDev = [y[i] for i in devIndices]
    
    return reviewsTrain, xTrain, yTrain, reviewsDev, xDev, yDev
    
def gridSearch(xTrain, yTrain, xDev, yDev, options):
    paramCombos = myProduct(options)
    bestCombo, bestCrossEntropy = None, float('inf')
    scores = {}
    
    for combo in paramCombos:
        cnn = CNN(numFilters=combo['numFilters'], windowSize=combo['windowSize'])
        cnn.fit(xTrain[:combo['numTrain']], yTrain[:combo['numTrain']], numEpochs=combo['numEpochs'],
                batchSize=combo['batchSize'], verbose=True)
        crossEntropy, accuracy = cnn.evaluate(xDev, yDev, showAccuracy=True)
        scores[tuple(combo.items())] = (crossEntropy, accuracy)
        if crossEntropy < bestCrossEntropy:
            bestCombo, bestCrossEntropy = combo, crossEntropy
        print 'Combo: {}, CE: {}, accuracy: {}'.format(combo, crossEntropy, accuracy)
    return scores