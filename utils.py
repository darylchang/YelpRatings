import numpy as np
import itertools

def safeLog(x, minval=1e-12):
    return np.log(x.clip(min=minval))

def crossEntropy(yTrue, yPred):
    return -np.sum(yTrue * safeLog(yPred))/len(yTrue)

def makeOneHot(stars, dim):
    a = np.zeros(dim)
    a[stars-1] = 1
    return a

def readOneHot(a):
    return np.argmax(a) + 1

def myProduct(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def softmax(x):
    max_elements = np.max(x, axis=1)
    x -= max_elements.reshape(x.shape[0], 1)
    
    exp = np.exp(x)
    sums = np.sum(exp, axis=1).reshape(x.shape[0], 1)
    x = exp / sums
    return x
