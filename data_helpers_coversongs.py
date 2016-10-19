import numpy as np
import itertools
import librosa
import random
import pickle
import os

def txt_to_cliques(shs_loc):
	'''
	creates a dictionary out of second hand songset 'cliques'
	or groups of cover songs and returns the dict of cliques
	based on their msd id
	'''
	shs = list(open(shs_loc))
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

def feature_extract(songfile_name):
	'''
	extracts features from song given a song file name
	**assumes working directory contains raw song files**
	returns a tuple containing songfile name and numpy array of song features
	'''
	#print(songfile_name)
	song_loc = os.path.abspath(songfile_name)
	y, sr = librosa.load(song_loc)
	desire_spect_len = 2580
	C = librosa.cqt(y=y, sr=sr, hop_length=512, fmin=None, 
					n_bins=84, bins_per_octave=12, tuning=None,
					filter_scale=1, norm=1, sparsity=0.01, real=True)
	# get power spectrogram
	C = librosa.logamplitude(C**2)
	# if spectral respresentation too long, crop it, otherwise, zero-pad
	if C.shape[1] >= desire_spect_len:
		C = C[:,0:desire_spect_len]
	else:
		C = np.pad(C,((0,0),(0,desire_spect_len-C.shape[1])), 'constant')
	return songfile_name, C

def create_feature_matrix(song_folder):
	feature_matrix = {}
	exceptions = []
	for filename in os.listdir(song_folder):
		if filename.endswith(".mp3"):
			try:
				name, features = feature_extract(filename)
				feature_matrix[name] = features
			except:
				exceptions.append(filename)
	return feature_matrix, exceptions

song_folder = '/scratch/mss460/shs/shs_train'
#hadoop song folder
hadoop_song_folder = '/user/mss460/shs/shs_train'
song_folder = '/Volumes/Amelia_Red_2TB/shs/shs_train'

def save_feature_matrix(song_folder):
	fm = create_feature_matrix(song_folder)
	fileHandle = open('/Volumes/Amelia_Red_2TB 1/shs/training_set_cqt.p', "wb")
	pickle.dump(fm, fileHandle)

def get_labels(cliques):
	# get and flatten all combination of coversongs
	positive_examples = (list(itertools.combinations(val,2)) for key,val in cliques.items())
	positive_examples = [i for j in positive_examples for i in j]
	positive_labels = [[1,0] for _ in positive_examples]
	# generate negative examples of an equivalent length to the positive examples list
	song_from_each_clique = (random.choice(val) for key,val in cliques.items())
	negative_examples = itertools.combinations(song_from_each_clique,2)
	negative_examples = list(itertools.islice(negative_examples, len(positive_examples)))
	negative_labels = [[0,1] for _ in negative_examples]

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
