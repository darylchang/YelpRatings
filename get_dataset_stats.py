import json
from collections import Counter

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

if __name__=='__main__':
	getStarDist()