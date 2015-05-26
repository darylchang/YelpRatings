import array, io, json, collections
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error
        
class SentimentClassifier:

	def __init__(self, cutoff=10000):
		self.wordVecs = self.readGlove()
		self.reviews, self.features, self.targets = self.readYelpReviews(cutoff)
		self.model = GaussianNB()

	# Adapted from maciejkula
	# Read in the GloVe vector data
	def readGlove(self):
		dct = {}
		with io.open('vectors/glove.6B.300d.txt', 'r', encoding='utf-8') as savefile:
			for i, line in enumerate(savefile):
				print i
				tokens = line.split(' ')

				word = tokens[0]
				entries = tokens[1:]

				dct[word] = np.array([float(x) for x in entries])
		return dct

	# Read in the Yelp review data
	def readYelpReviews(self, cutoff=None):
		reviews, features, targets = [], [], []
		with io.open('data/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as savefile:
		    for i, line in enumerate(savefile):
		    	print i
		    	if cutoff and i > cutoff:
		    		break
		    	review = json.loads(line)
		    	targets.append(review['stars'])
		    	features.append(self.generateFeatures(review))
		    	reviews.append(review)
		return reviews, features, targets

	def generateFeatures(self, review):
		text = review['text']
		featureVec = np.mean([self.wordVecs[token] for token in text.split() if token in self.wordVecs], axis=0)
		return featureVec

	def train(self, start, end):
		self.model.fit(self.features[start:end], self.targets[start:end])

	def test(self, start, end):
		predictions = self.model.predict(self.features[start:end])
		print mean_absolute_error(self.targets[start:end], predictions)

def main():
	clf = SentimentClassifier()
	clf.train(0, 10000)
	clf.test(10000, 15000)

if __name__ == "__main__":
	main()


