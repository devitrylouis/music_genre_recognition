import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys


#-------------------------------------------------------------------------------------------------------------------

def compute_melspectrogram(audio_path):

	y, sr = librosa.load(audio_path, sr=None, mono = True)
	S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

	plt.figure(figsize=(10, 4))
	librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Mel spectrogram')
	plt.tight_layout()
	plt.show()

#--------------------------------------------------------------------------------------------------------------------
"""
def compute_spectrogram_trial(audio_path):

	y, sr = librosa.load(audio_path, sr=None, mono = True)

	if sr != 44100:
		y = librosa.resample(y, sr, 44100)
		sr = 44100

	mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc = 20)

	mfcc = librosa.amplitude_to_db(mfcc)

	# normalization
	mfcc_norm = mfcc - mfcc.min()
	mfcc_norm /= mfcc.max()

	Cov_norm = np.cov(mfcc_norm)

	print("Every sample will have a matrix of {}".format(mfcc.shape))

	plt.figure(figsize=(20, 12))
	librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='hz')
	plt.ylabel("Freq")
	plt.xlabel("Time")
	plt.title("MFCC of the song", fontsize=15)
	plt.show()

	plt.figure(figsize=(12, 12))
	plt.imshow(Cov_norm, extent=[0, 20, 0, 20], aspect='auto');
	plt.ylabel("Amplitude")
	plt.xlabel("Time")
	plt.title("Covariance Matrix of the Song", fontsize=15)
	plt.colorbar()
	plt.show()

"""
#---------------------------------------------------------------------------------------------------------------------

def get_audio_path(audio_dir, track_id):
    
    tid_str = '{:06d}'.format(track_id)

    path = os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

    if os.path.isfile(path):
    	return path
    else:
    	return ' '

#-----------------------------------------------------------------------------------------------------------------------

def main():

	track_id_file = '/home/ayush/FOML_Project/track_id_all.csv'
	fma_home = '/home/ayush/FOML_Project/fma_small'

	track_id_to_parent_genre_title_path = '/home/ayush/FOML_Project/track_id_to_parent_genre_title.json'
	genre_title_to_genre_id_path = '/home/ayush/FOML_Project/genre_title_to_genre_id.json'

	track_data_frame = pd.read_csv(track_id_file,header=None)
	track_data_frame.columns = ['track_id']
	
	track_path = [get_audio_path(fma_home,x) for x in list(track_data_frame['track_id'])]
	track_data_frame['track_path'] = track_path

	track_data_frame = track_data_frame[track_data_frame['track_path']!=' ']
	track_data_frame.to_csv('track_id_to_audio_path_fma_small.csv',index=False,sep=',')

	compute_melspectrogram(track_data_frame['track_path'][0])

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	main()