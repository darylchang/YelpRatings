from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adadelta
import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error, accuracy_score

class CNN:
    
    # Create one filter that passes over bigrams by default
    def __init__(self, numFilters=1, windowSize=2, minLen=10.):
        self.model = Sequential()
        self.numFilters = numFilters
        self.windowSize = windowSize
        self.numClasses = 5
        self.minLen = minLen
    
    def fit(self, xTrain, yTrain, numEpochs=10, batchSize=15, dropout=0.5, l=0.01, lr=0.01, padded=False, verbose=False):
        if padded:
            # Zero pad matrices
            maxLen = max([matrix.shape[1] for matrix in xTrain])
            d = xTrain[0].shape[2]
            for matrix in xTrain:
                matrix.resize((1,maxLen,d), refcheck=False)
            xTrain = np.array(xTrain)
        else:
            # Split into chunks
            d = xTrain[0].shape[2]
            xTrainOld, yTrainOld = list(xTrain), list(yTrain)
            xTrain, yTrain = np.empty((0,1,self.minLen,d), float), np.empty((0,1), int)
            
            for matrix, label in zip(xTrainOld, yTrainOld):
                numChunks = int(math.ceil(matrix.shape[1] / self.minLen))
                subMatrices = [np.array(subMatrix) for subMatrix in np.array_split(matrix[0], numChunks)]
                for subMatrix in subMatrices:
                    subMatrix.resize((1,self.minLen,d),refcheck=False)
                subMatrices = np.array(subMatrices)
                xTrain = np.vstack((xTrain, subMatrices))
                newLabels = np.array([[label] for i in range(numChunks)])
                yTrain = np.vstack((yTrain, newLabels))
        
        # Randomize training example order
        shuffled = zip(xTrain, yTrain)
        random.shuffle(shuffled)
        xTrain, yTrain = [np.array(t) for t in zip(*shuffled)]
        
        # Get dimensions
        batchSize, stackSize, numWords, d = xTrain.shape
        numPoolRows = numWords - self.windowSize + 1
        print numPoolRows

        # Create convolutional neural net and fit data
        self.model.add(Convolution2D(self.numFilters, stackSize, self.windowSize, d, border_mode='full')) 
        self.model.add(MaxPooling2D(poolsize=(numPoolRows, 1), ignore_border=False))
        self.model.add(Flatten())
        self.model.add(Dropout(dropout))
        self.model.add(Dense(self.numFilters, 1, W_regularizer=l2(l)))
        self.model.add(Activation('relu'))
        sgd = Adadelta(lr=lr)
        self.model.compile(loss='mse', optimizer=sgd)

        self.model.fit(xTrain, yTrain, nb_epoch=numEpochs, batch_size=batchSize, verbose=verbose)
        
    def evaluate(self, xTest, yTest, showAccuracy=False, padded=False, chunked=False, verbose=False):
        # Zero pad matrices
        if padded:
            maxLen = max([matrix.shape[1] for matrix in xTest])
            d = xTest[0].shape[2]
            for matrix in xTest:
                matrix.resize((1,maxLen,d), refcheck=False)
            xTest = np.array(xTest)
            
            regressionScore = self.model.evaluate(xTest, yTest, verbose=verbose)
            regressionPredictions = self.model.predict(xTest)
            classPredictions = [[round(prediction)] for prediction in regressionPredictions]
            matches = [a==b for a,b in zip(yTest, classPredictions)]
            accuracy = np.sum(matches) / float(len(matches))
            return regressionScore, accuracy, regressionPredictions, classPredictions
            
        # Split into chunks
        elif chunked:
            
            d = xTest[0].shape[2]
            regPred, yPred = np.empty((0,1), float), np.empty((0,1), int)
            
            for matrix in xTest:
                numChunks = int(math.ceil(matrix.shape[1] / self.minLen))
                subMatrices = [np.array(subMatrix) for subMatrix in np.array_split(matrix[0], numChunks)]
                for subMatrix in subMatrices:
                    subMatrix.resize((1,self.minLen,d),refcheck=False)
                subMatrices = np.array(subMatrices)
                reg = np.mean(self.model.predict(subMatrices, verbose=verbose))
                regPred = np.vstack((regPred, reg))
                yPred = np.vstack((yPred, round(reg)))
            
            regressionScore = mean_squared_error(yTest, regPred)
            accuracy = accuracy_score(yTest, yPred)
            
            return regressionScore, accuracy, regPred, yPred
        
        # Predict document by document
        else:
            for matrix in xTest:
                matrix = np.array([matrix])
                regressionPrediction = self.model.predict(matrix)
                print regressionPrediction
                
        
        