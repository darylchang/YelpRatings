import array, io, json, collections
import numpy as np

def generateFeatures(gloveVecs, reviews):
    features = []
    for review in reviews:
        text = review['text']
        featureVec = np.mean([gloveVecs[token] for token in text.split() if token in gloveVecs], axis=0)
        if featureVec.size and not np.any(np.isnan(featureVec)):
            features.append(featureVec)
        else:
            features.append(np.zeros(features[0].shape[0]))   
    return np.array(features)

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
def readYelpReviews(cutoff=None):
    reviews, labels = [], []
    with io.open('data/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(savefile):
            if i%100==0:
                print i
            if cutoff and i > cutoff:
                break
            review = json.loads(line)
            reviews.append(review)
            labels.append(review['stars'])
    return np.array(reviews), np.array(labels)