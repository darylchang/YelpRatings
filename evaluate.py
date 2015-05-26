from sklearn.metrics import mean_absolute_error
import numpy as np

def safeLog(x, minval=1e-12):
    return np.log(x.clip(min=minval))

def crossEntropy(yTrue, yPred):
    return -np.sum(yTrue * safeLog(yPred))

def makeOneHot(label, dim):
    a = np.zeros(dim)
    a[label] = 1
    return a

def evaluateDev(classifier, reviews, x, y, trainPercentage=70, devPercentage=15):
    # Get dimensions
    M = x.shape[0]
    N = x.shape[1]
    numClasses = len(np.unique(y))
    
    # Set train-dev-test split indices
    trainLower, trainUpper = 0, int(trainPercentage * M / 100.)
    devLower, devUpper = trainUpper, trainUpper + int(devPercentage * M / 100.) 
    
    xTrain, yTrain = x[trainLower:trainUpper], y[trainLower:trainUpper]
    xDev, yDev = x[devLower:devUpper], y[devLower:devUpper]
    classifier.fit(xTrain, yTrain)
    yPred = classifier.predict_proba(xDev)
    yTrue = [makeOneHot(label-1,numClasses) for label in yDev]
    print "Cross Entropy on dev set: {}".format(crossEntropy(yTrue, yPred))
    