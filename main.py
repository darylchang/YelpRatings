import array, io, json, collections
import numpy as np

dct = {}
        
# Adapted from maciejkula
# Read in the GloVe vector data
def readGlove():
	with io.open('vectors/glove.6B.300d.txt', 'r', encoding='utf-8') as savefile:
		for i, line in enumerate(savefile):
			print i
			tokens = line.split(' ')

			word = tokens[0]
			entries = tokens[1:]

			dct[word] = np.array([float(x) for x in entries])
	return dct

# Read in the Yelp review data
def readYelpReviews():
	reviews = []
	with io.open('data/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as savefile:
	    for i, line in enumerate(savefile):
	    	print i
	    	review = json.loads(line)
	    	reviews.append(review)
	return reviews

print len(readYelpReviews())

