from sklearn.metrics import mean_absolute_error
import numpy as np

def crossEntropy(yTrue, yPred):
    return -np.sum(yTrue * np.log(yPred))

def evaluateDev(classifier, reviews, x, y, trainPercentage=70, devPercentage=15):
    # Set train-dev-test split indices
    N = x.shape[0]
    trainLower, trainUpper = 0, int(trainPercentage * N / 100.)
    devLower, devUpper = trainUpper, trainUpper + int(devPercentage * N / 100.) 
    
    xTrain, yTrain = x[trainLower:trainUpper], y[trainLower:trainUpper]
    xDev, yDev = x[devLower:devUpper], y[devLower:devUpper]
    classifier.fit(xTrain, yTrain)
    yPred = classifier.predict(xDev)
    print "Cross Entropy on dev set: {}".format(crossEntropy(yDev, yPred))
    