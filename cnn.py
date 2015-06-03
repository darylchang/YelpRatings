from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adadelta
import numpy as np

class CNN:
    
    # Create one filter that passes over bigrams by default
    def __init__(self, numFilters=1, windowSize=2):
        self.model = Sequential()
        self.numFilters = numFilters
        self.windowSize = windowSize
        self.numClasses = 5
    
    def fit(self, xTrain, yTrain, numEpochs=10, batchSize=15, dropout=0.5, l=0.5, verbose=False):
        batchSize, stackSize, numWords, d = xTrain.shape
        numPoolRows = numWords - self.windowSize + 1
        
        self.model.add(Convolution2D(self.numFilters, stackSize, self.windowSize, d)) 
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(poolsize=(numPoolRows, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(self.numFilters, 1, W_regularizer=l2(l)))
        self.model.add(Dropout(dropout))
        self.model.add(Activation('relu'))
        sgd = Adadelta()
        self.model.compile(loss='mse', optimizer=sgd)
        
        # Remove rows of zeros in xTrain
        #xTrain = xTrain[np.any(xTrain, axis=3)]
        self.model.fit(xTrain, yTrain, nb_epoch=numEpochs, batch_size=batchSize, verbose=verbose)
        
    def evaluate(self, xTest, yTest, showAccuracy=False, verbose=False):
        # Remove rows of zeros in xTest
        #xTest = xTest[np.any(xTrain, axis=3)]
        score = self.model.evaluate(xTest, yTest, show_accuracy=showAccuracy, verbose=verbose)
        return score
    
    def predict(self, xTest):
        return self.model.predict(xTest)
    
    def predictClasses(self, xTest):
        return self.model.predict_classes(xTest)