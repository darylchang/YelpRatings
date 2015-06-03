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
def readYelpReviews(numPerLabel=None, useOneHot=False):
    reviews, labels = [], []
    labelCounts = {i+1:0 for i in range(NUM_CLASSES)}
    
    # Read in an equal number of each class
    with io.open('data/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(savefile):
            # Output progress
            if i%100==0:
                print i
            
            # Load review and add to list if numPerLabel for the label is not yet reached
            review = json.loads(line)
            label = review['stars']
            if all([count >= numPerLabel for count in labelCounts.values()]):
                break
            elif labelCounts[label] >= numPerLabel:
                continue
            else:
                labelCounts[label] += 1
                label = makeOneHot(label,NUM_CLASSES) if useOneHot else label
                reviews.append(review)
                labels.append(label)
  
    return np.array(reviews), np.array(labels)