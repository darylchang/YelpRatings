from imdb import *

path="imdb.pkl"
path = get_dataset_file(
        path, "imdb.pkl",
        "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")

if path.endswith(".gz"):
    f = gzip.open(path, 'rb')
else:
    f = open(path, 'rb')

train_set = cPickle.load(f)
test_set = cPickle.load(f)
f.close()

valid_portion=0.1
n_words=100000

# split training set into validation set
train_set_x, train_set_y = train_set
n_samples = len(train_set_x)
sidx = numpy.random.permutation(n_samples)
n_train = int(numpy.round(n_samples * (1. - valid_portion)))
valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
train_set_x = [train_set_x[s] for s in sidx[:n_train]]
train_set_y = [train_set_y[s] for s in sidx[:n_train]]

train_set = (train_set_x, train_set_y)
valid_set = (valid_set_x, valid_set_y)

def remove_unk(x):
    return [[1 if w >= n_words else w for w in sen] for sen in x]

test_set_x, test_set_y = test_set
valid_set_x, valid_set_y = valid_set
train_set_x, train_set_y = train_set

train_set_x = remove_unk(train_set_x)
valid_set_x = remove_unk(valid_set_x)
test_set_x = remove_unk(test_set_x)

train = (train_set_x, train_set_y)
valid = (valid_set_x, valid_set_y)
test = (test_set_x, test_set_y)

# list of 25000 lists of ints
review = ''
for i in test_set_x[0]:
	review = review + 