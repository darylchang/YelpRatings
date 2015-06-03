import array, io, json, collections
from utils import *
import numpy as np
from collections import defaultdict
import random

NUM_CLASSES = 5

# Adapted from maciejkula
# Read in the GloVe vector data
def readGlove():
    dct = {}
    with io.open('vectors/glove.6B.300d.txt', 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(savefile):
            if i%100==0:
                print i
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]
            dct[word] = np.array([float(x) for x in entries])
    return dct

# Read in the Yelp review data
def readYelpReviews(numTotal=None, numPerLabel=None, useOneHot=False, balanced=False):
    reviews, labels = [], []
    labelCounts = {i+1:0 for i in range(NUM_CLASSES)}
    
    with io.open('data/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(savefile):
            # Output progress
            if i%100==0:
                print i
            
            # Reached threshold for total number of training examples
            if numTotal and i == numTotal:
                break
            
            # Load review
            review = json.loads(line)
            label = review['stars']
            
            # If balanced is True, read in an equal number of each class
            if balanced:
                if all([count >= numPerLabel for count in labelCounts.values()]):
                    break
                elif balanced and labelCounts[label] >= numPerLabel:
                    continue
      
            labelCounts[label] += 1
            label = makeOneHot(label,NUM_CLASSES) if useOneHot else label
            reviews.append(review)
            labels.append(label)
  
    return np.array(reviews), np.array(labels)