import numpy as np
import librosa
import os
import pickle
import gzip

def feature_extract(songfile_name):
	'''
	takes: filename
	outputs: audio feature representation from that file (currently cqt)
	**assumes working directory contains raw song files**
	returns a tuple containing songfile name and numpy array of song features
	'''
	song_loc = os.path.abspath(songfile_name)
	y, sr = librosa.load(song_loc)
	desire_spect_len = 2580
	C = librosa.cqt(y=y, sr=sr, hop_length=512, fmin=None,
					n_bins=84, bins_per_octave=12, tuning=None,
					filter_scale=1, norm=1, sparsity=0.01, real=False)
    # get power spectrogram
    C = librosa.logamplitude(C**2)
    # if spectral respresentation too long, crop it, otherwise, zero-pad
    if C.shape[1] >= desire_spect_len:
        C = C[:,0:desire_spect_len]
	else:
		C = np.pad(C,((0,0),(0,desire_spect_len-C.shape[1])), 'constant')
	return songfile_name, C

def create_feature_matrix(song_folder):
	'''
	takes: song folder filled with mp3 files as input
	outputs: key-value pairs of filenames : feature representations (spectrograms)
    list of filenames that caused exceptions
	'''
	feature_matrix = {}
	exceptions = []
	for filename in os.listdir(song_folder):
		if filename.endswith(".mp3"):
			try:
				print("Processing: ", filename, end="\r")
				name, features = feature_extract(os.path.join(song_folder,filename))
				feature_matrix[name] = features
			except:
				print("Exception on: ", filename)
				exceptions.append(filename)
	return feature_matrix, exceptions

def save_feature_matrix(song_folder,save_path):
	fm,excepts = create_feature_matrix(song_folder)
	fileHandle = gzip.open(save_path, "wb")
	pickle.dump(fm, fileHandle)
    fileHandle.close()

song_folder = '/scratch/mss460/shs/shs_train'
save_path = '/home/mss460/training_set_cqt.pickle'
save_feature_matrix(song_folder,save_path)

