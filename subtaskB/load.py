def parse_dataset(dataset):
    '''Loads the dataset .txt file with label-tweet on each line and parses the dataset.'''
    y = []
    corpus = []
    dataset_name = dataset.lower()
    with open(dataset, 'r') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"):	# discard first line if it contains metadata
                line = line.rstrip()	# remove trailing whitespace
                if ("train" in dataset_name) or ("gold_test" in dataset_name):
                    label = int(line.split("\t")[1])
                    tweet = line.split("\t")[2]
                    y.append(label)
                else:
                   tweet = line.split("\t")[1]
                corpus.append(tweet)
    if ("train" in dataset_name) or ("gold_test" in dataset_name):
        return corpus, y
    else:
       return corpus