import json
from collections import Counter
from sets import Set

stars = Counter()
positive = Set()
negative = Set()

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

# with open('data/yelp_academic_dataset_review.json') as data_file:    
#     for line in data_file:
#     	data = json.loads(line)
#     	stars[data["stars"]] += 1

# with open('stats/stars.txt', 'w+') as f:
# 	f.write("Counts for star ratings in reviews\n")
# 	for k,v in stars.most_common():
# 		f.write("Rating {}: {}\n".format(k,v))