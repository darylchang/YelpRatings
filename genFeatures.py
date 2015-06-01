import numpy as np

# For use with sklearn classifiers; not yet adapted for Keras
# Calculates mean of words vectors in review
def genMeanFeatures(gloveVecs, reviews):
    features = []
    for review in reviews:
        text = review['text']
        featureVec = np.mean([gloveVecs[token] for token in text.split() if token in gloveVecs], axis=0)
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
        featureMatrix = np.array([[gloveVecs[token] for token in text.split() if token in gloveVecs]])
        if featureMatrix.size and not np.any(np.isnan(featureMatrix)):
            features.append(featureMatrix)
        else:
            features.append(np.array([np.zeros(features[0].shape)]))  
    
    # Reformat for Keras
    maxLen = max([matrix.shape[1] for matrix in features])
    d = features[0].shape[2]
    for matrix in features:
        matrix.resize((1,maxLen,d), refcheck=False)
    features = np.array(features)
    return np.array(features)
