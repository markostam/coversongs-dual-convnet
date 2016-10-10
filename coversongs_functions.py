def createTimeWarpBeatMatched(ogPath,coverPath):
	import librosa
	import numpy as np
	from scipy.spatial.distance import euclidean
	from fastdtw import fastdtw

	sr = 16000
	
	#import the audio
	og, sr = librosa.load(ogPath)
	cover, sr = librosa.load(coverPath,sr)
	#extract the beats
	ogTempo, ogBeats = librosa.beat.beat_track(y=og, sr=sr)
	coverTempo, coverBeats = librosa.beat.beat_track(y=cover, sr=sr)
	#extract the CQT
	ogCQT = librosa.cqt(og, sr=sr)
	coverCQT = librosa.cqt(cover, sr=sr)
	#TODO: check for 0 energy
	#TODO: log CQT
	#downsample CQT to beats
	ogSync = librosa.feature.sync(ogCQT, ogBeats)
	coverSync = librosa.feature.sync(coverCQT, coverBeats)
	#get DTW distance
	distance, path = fastdtw(ogSync.transpose(), coverSync.transpose(), dist=euclidean)
	#stretched indices of cover mapped to original
	coverIndices = [j[1] for j in path]
	#align cover to original
	coverAligned = [coverSync.transpose()[i] for i in coverIndices]
	coverAligned = coverAligned[0:len(ogSync.transpose())]
	coverAligned = np.asarray(coverAligned).transpose()

	return (ogSync,coverAligned)

def createShiftedSpectrogram(path):
	sr = 16000