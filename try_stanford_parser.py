import os
import read

from nltk.parse import stanford
import nltk.data

os.environ['STANFORD_PARSER'] = '/Users/AprilYu/Desktop/cs224d/YelpRatings/parser/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/Users/AprilYu/Desktop/cs224d/YelpRatings/parser/stanford-parser-3.5.2-models.jar'

parser = stanford.StanfordParser(model_path="/Users/AprilYu/Desktop/cs224d/YelpRatings/parser/stanford-parser-3.5.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

reviews, targets = read.readYelpReviews(cutoff=10)

review_text = []
for r in reviews:
	review_text.append(r['text'])

for re in review_text:
	sent = tokenizer.tokenize(re)
	print sent
	sentences = parser.raw_parse_sents(tuple(sent))
	# GUI
	for line in sentences:
		for sentence in line:
			print sentence