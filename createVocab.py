import read
from collections import Counter

reviews, labels = read.readYelpReviews(numPerLabel=1)

text = []
for r in reviews:
    text.append(r['text'])

one_string = " ".join(text)
words = one_string.split()

counts = Counter()
for w in words:
	w = w.strip('\'-,.:;!?)(').lower()
	counts[w] += 1

vocab = sorted(counts.keys(), key=counts.get, reverse=True)

with open('yelp.vocab', 'w') as f:
	for w in vocab:
		f.write(w+'\n')