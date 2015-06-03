import read

def prepare_data(seqs, labels, maxlen=None):
    pass

def load_data(path="imdb.pkl", n_words=100000, valid_portion=0.15, maxlen=None,
              sort_by_len=True):
    reviews, labels = read.readYelpReviews(numPerLabel=30)

    text = []
    for r in reviews:
        text.append(r['text'])

    

    # return train, valid, test