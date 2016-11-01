import numpy as np
import itertools
import random
import gzip
import pickle
import os
import pdb

def txt_to_cliques(shs_loc):
    '''
    creates a dictionary out of second hand songset 'cliques'
    or groups of cover songs and returns the dict of cliques
    based on their msd id
    '''
    shs = list(open(shs_loc, encoding = 'utf-8'))
    shs = shs[14:]
    cliques = {}
    for ent in shs:
        ent = ent.replace('\n','')
        if ent[0] == '%':
            tempKey = ent.lower()
            cliques[tempKey] = []
        else:
            cliques[tempKey].append(ent.split("<SEP>")[0]+'.mp3')
    return cliques

def get_labels(cliques):
    # get and flatten all combination of coversongs
    positive_examples = (list(itertools.combinations(val,2)) for key,val in cliques.items())
    positive_examples = [i for j in positive_examples for i in j]
    positive_len_og = len(positive_examples)
    positives_flipped = [(i[1],i[0]) for i in positive_examples]
    positive_examples += positives_flipped
    positive_labels = [[0,1] for _ in positive_examples]
    # generate negative examples of an equivalent length to the positive examples list
    song_from_each_clique = (random.choice(val) for key,val in cliques.items())
    negative_examples = itertools.combinations(song_from_each_clique,2)
    negative_examples = list(itertools.islice(negative_examples, positive_len_og))
    negatives_flipped = [(i[1],i[0]) for i in negative_examples]
    negative_examples += negatives_flipped
    negative_labels = [[1,0] for _ in negative_examples]

    x = positive_examples + negative_examples
    y = positive_labels + negative_labels
    return x,y


def get_labels_for_samesong_test(cliques):
    '''
    test function that generates positive examples/labels for songs where positive
    is just the same song and negative is different song
    '''
    # positive examples are the same songs
    positive_examples = [(i,i) for j in cliques.values() for i in j]
    positive_len_og = len(positive_examples)
    positive_labels = [[0,1] for _ in positive_examples]
    # negative examples are any other song combination
    song_from_each_clique = (random.choice(val) for key,val in cliques.items())
    negative_examples = itertools.combinations(song_from_each_clique,2)
    negative_examples = list(itertools.islice(negative_examples, positive_len_og))
    negative_labels = [[1,0] for _ in negative_examples]

    x = positive_examples + negative_examples
    y = positive_labels + negative_labels
    return x,y

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffled_data = np.random.permutation(data)
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def read_from_pickles(path_to_pickles):
    '''
    function that loads a dictionary of filename : cqt matrix
    from a directory of gzipped pickles containing the chunked data
    '''
    spect_dict = {}
    for file in os.listdir(path_to_pickles):
        if file.endswith('.pickle.gz'):
            with gzip.open(os.path.join(path_to_pickles,file),'rb') as f:
                temp_dict = pickle.load(f, encoding='latin-1')
            spect_dict.update(temp_dict)
    return spect_dict

def prune_cliques(cliques,spect_dict):
    '''
    prunes the coversong cliques wrt the spectrogram dictionary
    in case any spectrograms don't exist due to data loss e.g.
    scraping errors, filesize=0, etc...
    '''
    pruned_cliques={}
    for clique_key, clique_val in cliques.items():
        pruned_cliques[clique_key]=[]
        for songid in clique_val:
            if songid in spect_dict.keys():
                pruned_cliques[clique_key].append(songid)
        if len(pruned_cliques[clique_key]) < 2:
            del pruned_cliques[clique_key]
    return pruned_cliques

def cliques_to_dev_train(cliques,percent_dev):
    '''
    splits all cliques into dev/train sets so as to not have
    overlapping songs in dev/train to prevent overfitting
    '''
    dev_len = int(len(cliques)*percent_dev)
    cliques_list = list(cliques.items())
    train_cliques = dict(cliques_list[:-dev_len])
    dev_cliques = dict(cliques_list[-dev_len:])
    return train_cliques, dev_cliques

def randomly_shuffle_xy_data(x,y):
    # Randomly shuffle data
    np.random.seed(420)
    if len(x) != len(y):
        raise Exception
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x,y = np.array(x), np.array(y)
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled, y_shuffled

def binarize_cliques(cliques):
    '''
    removes any more than two songs per clique to prevent overfitting
    '''
    binary_cliques={}
    for clique_key, clique_val in cliques.items():
        binary_cliques[clique_key]=random.sample(cliques[clique_key],2)
    return binary_cliques
