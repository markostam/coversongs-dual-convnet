import numpy as np
import librosa
import os

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
					filter_scale=1, norm=1, sparsity=0.01, real=False)
	# if spectral respresentation too long, crop it, otherwise, zero-pad
	if C.shape[1] >= desire_spect_len:
		C = C[:,0:desire_spect_len]
	else:
		C = np.pad(C,((0,0),(0,desire_spect_len-a.shape[1])), 'constant')
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

def save_feature_matrix(song_folder,save_path):
	fm = create_feature_matrix(song_folder)
	fileHandle = open(save_path, "wb")
	pickle.dump(fm, fileHandle)

song_folder = '/scratch/mss460/shs/shs_train'
save_path = '/scratch/mss460/shs/training_set_cqt.pickle'
save_feature_matrix(song_folder,save_path)

