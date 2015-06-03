import numpy as np
from nltk.tokenize import RegexpTokenizer

pattern = r'''(?x)           # set flag to allow verbose regexps
                      ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
                      | \-?[\d\w]+([-']\w+)*    # words w/ optional internal hyphens/apostrophe
                      | \$?\d+(\.\d+)?%? # numbers, incl. currency and percentages
                      | [+/\-@&*]        # special characters with meanings
            '''
tokenizer = RegexpTokenizer(pattern)

# For use with sklearn classifiers; not yet adapted for Keras
# Calculates mean of words vectors in review
def genMeanFeatures(gloveVecs, reviews):
    features = []
    for review in reviews:
        text = review['text']
        featureVec = np.mean([gloveVecs[token] for token in tokenizer.tokenize(text) if token in gloveVecs], axis=0)
        if featureVec.size and not np.any(np.isnan(featureVec)):
            features.append(featureVec)
        else:
            features.append(np.zeros(features[0].shape[0]))   
    return np.array(features)

# Formatted for Keras classifiers
def genKerasFeatures(gloveVecs, reviews):
    features = []
    for review in reviews:
        text = review['text']
        featureMatrix = np.array([[gloveVecs[token] for token in tokenizer.tokenize(text) if token in gloveVecs]])
        if featureMatrix.size and not np.any(np.isnan(featureMatrix)):
            features.append(featureMatrix)
        else:
            features.append(np.array([np.zeros(features[0].shape)]))  
    return features