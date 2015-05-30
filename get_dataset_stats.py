import json
from collections import Counter
from sets import Set
from nltk.tokenize import RegexpTokenizer
from matplotlib import pyplot as plt
import numpy as np

starDist = Counter({5: 579527, 4: 466599, 3: 222719, 1: 159812, 2: 140608})

def plotBarChart(counter, xlabel, ylabel, title):
	labels, values = zip(*counter.items())
	indexes = np.arange(1, 6)
	width = 1

	plt.bar(indexes, values, width)
	plt.xticks(indexes + width * 0.5, labels)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

def getLexicon():
	positive, negative = Set(), Set()

	with open('lexicon/negative-words.txt') as neg:
		for l in neg:
			if l[0] == ';':
				continue
			negative.add(l.strip('\n'))

	with open('lexicon/positive-words.txt') as neg:
		for l in neg:
			if l[0] == ';':
				continue
			positive.add(l.strip('\n'))

	return positive, negative

def getStarDist():
	stars = Counter()
	with open('data/yelp_academic_dataset_review.json') as data_file:    
	    for index, line in enumerate(data_file):
	    	data = json.loads(line)
	    	stars[data["stars"]] += 1

	with open('stats/stars.txt', 'w+') as f:
		f.write("Counts for star ratings in reviews\n")
		for k,v in stars.most_common():
			f.write("Rating {}: {}\n".format(k,v))

def getSentiment():
	posWords, negWords = getLexicon()
	posWordDist, negWordDist = Counter(), Counter()
	with open('data/yelp_academic_dataset_review.json') as data_file:    
	    for index, line in enumerate(data_file):
	    	# Print out indices
	    	if index % 1000 == 0:
	    		print index

	    	# Tokenize data
	    	data = json.loads(line)
	    	tokenizer = RegexpTokenizer(r'\w+')
        	tokens = tokenizer.tokenize(data["text"])
	    	numWords = len(tokens)

	    	if numWords:
		    	# Calculate percentage of positive and negative words
		    	numPosWords = sum([1. for _ in tokens if _ in posWords])
		    	numNegWords = sum([1. for _ in tokens if _ in negWords])
		    	posWordDist[data["stars"]] += numPosWords / numWords
		    	negWordDist[data["stars"]] += numNegWords / numWords

	# Normalize by number of reviews in each ratings bucket
	for k,v in posWordDist.items():
		posWordDist[k] /= starDist[k]
		negWordDist[k] /= starDist[k]

	# Write to file
	with open('stats/sentiment.txt', 'w+') as f:
		for k,v in posWordDist.items():
			f.write("Average percentage of positive words for {}-star reviews: {}\n".format(k,v))
		for k,v in negWordDist.items():
			f.write("Average percentage of negative words for {}-star reviews: {}\n".format(k,v))

if __name__=='__main__':
	#getSentiment()
	cPos = Counter({5: 0.0745142560276365, 4: 0.06684858807379605, 3: 0.05375041703131739, 2: 0.04089520536026867, 1: 0.027294962988517392})
	cNeg = Counter({1: 0.034665701348088086, 2: 0.02832118569391694, 3: 0.020923946692072627, 4: 0.015263962841940525, 5: 0.012279220653938253})
	plotBarChart(cPos, 'Stars', 'Pos Words as Proportion of Text', 'Distribution of Positive Words in Yelp Reviews')
	plotBarChart(cNeg, 'Stars', 'Neg Words as Proportion of Text', 'Distribution of Negative Words in Yelp Reviews')

