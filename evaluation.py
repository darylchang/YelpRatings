from sklearn.metrics import mean_absolute_error
import numpy as np
import itertools
from cnn import CNN
import random
from utils import *

def trainDevSplit(reviews, x, y, trainPercentage=70, devPercentage=15):
    # Get dimensions
    M = x.shape[0]
    N = x.shape[1]
    y = np.argmax(y, axis=1)
    numClasses = len(np.unique(y))
    numTrainPerClass = int(trainPercentage / 100. * M) / numClasses
    numDevPerClass = int(devPercentage * M / 100.) / numClasses    
    trainCounts = {i:0 for i in range(numClasses)}
    devCounts = {i:0 for i in range(numClasses)}
    trainData, devData = [], []
    
    for i in range(M):
        currReview, currX, currY = reviews[i], x[i], y[i]
        if trainCounts[currY] < numTrainPerClass:
            trainData.append((currReview, currX, currY))
            trainCounts[currY] += 1
        elif devCounts[currY] < numDevPerClass:
            devData.append((currReview, currX, currY))
            devCounts[currY] += 1
        else:
            continue
    
    # Unzip tuples
    reviewsTrain, xTrain, yTrain = zip(*trainData)
    reviewsDev, xDev, yDev = zip(*devData)
    
    # Make y into one hot vectors again and x into a numpy array
    yTrain = [makeOneHot(label, numClasses) for label in yTrain]
    yDev = [makeOneHot(label, numClasses) for label in yDev]
    xTrain = np.array(xTrain)
    xDev = np.array(xDev)
    
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