# Utility Functions for the project
import numpy as np 
import pandas as pd
import json

#---------------------------------------------------------------------------------------------------------------------------------------

# Create Mapping from Genre Title to Genre ID
def map_genre_title_to_genre_id(genres_file):

	genres_data = pd.read_csv(genres_file,delimiter=',',index_col=0,header=0)
	genres_data = genres_data.reset_index()

	genre_id_list = list(genres_data['genre_id'])
	genre_title_list = list(genres_data['title'])

	dict_genre_title_to_genre_id = {}

	for i in range(len(genre_title_list)):
		dict_genre_title_to_genre_id[genre_title_list[i]] = genre_id_list[i]

	with open('genre_title_to_genre_id.json', 'w') as fp:
		json.dump(dict_genre_title_to_genre_id, fp, sort_keys=True, indent=4)

	return dict_genre_title_to_genre_id

#--------------------------------------------------------------------------------------------------------------------------------------

# Create Mapping from Track ID to Parent Genre Title
def map_track_id_to_parent_genre_title(tracks_file):

	tracks_data = pd.read_csv(tracks_file,delimiter=',',index_col=0,header=[0,1])

	tracks_data = tracks_data.reset_index()
	track_id_list = list(tracks_data['track_id'])
	parent_genre_title_list = list(tracks_data['track']['genre_top'])

	dict_track_id_to_parent_genre_title = {} 

	for i in range(len(track_id_list)):
		dict_track_id_to_parent_genre_title[track_id_list[i]] = parent_genre_title_list[i]

	with open('track_id_to_parent_genre_title.json', 'w') as fp:
		json.dump(dict_track_id_to_parent_genre_title, fp, sort_keys=True, indent=4)

	return dict_track_id_to_parent_genre_title

#--------------------------------------------------------------------------------------------------------------------------------------
# Process Echonest Tracks
def process_echonest_tracks(echonest_file,dict_genre_title_to_genre_id,dict_track_id_to_parent_genre_title):

	echonest_data = pd.read_csv(echonest_file,delimiter=',',index_col=0,header=[0,1,2])
	echonest_data = echonest_data.reset_index()
	#print len(echonest_tracks_ids)

	# Create Echonest DataFrame with headers track_id, parent_genre_id, parent_genre_title, echonest features, librosa_features
	echonest_labels = pd.DataFrame(columns=['track_id','parent_genre_id','parent_genre_title'])
	echonest_labels['track_id'] = list(echonest_data['track_id'])


	parent_genre_title_list = []
	parent_genre_id_list = []

	#Get Parent Title for Echonest
	for id in echonest_labels['track_id']:
		parent_genre_title_list.append(dict_track_id_to_parent_genre_title[id])

	echonest_labels['parent_genre_title'] = parent_genre_title_list

	#Get Parent Genre ID for Echonest
	for genre_title in parent_genre_title_list:
		if pd.isnull(genre_title):
			parent_genre_id_list.append(np.nan)
		else:
			parent_genre_id_list.append(dict_genre_title_to_genre_id[genre_title])

	echonest_labels['parent_genre_id'] = parent_genre_id_list

	echonest_labels.to_csv('echonest_labels.csv',sep=',',index=False)

#----------------------------------------------------------------------------------------------------------------------------------------
def merge_echonest_librosa_features(echonest_labels,echonest_file,librosa_file):

	# For tracks in echonest full data merge the corresponding echonest features and librosa features

	echonest_labels_data = pd.read_csv(echonest_labels,sep=',',index_col=0)
	#print echonest_labels_data.head()
	echonest_data = pd.read_csv(echonest_file,delimiter=',',index_col=0,header=[0,1,2])
	librosa_data = pd.read_csv(librosa_file,delimiter=',',index_col=0,header=[0,1,2])

	echonest_full_data = librosa_data.merge(echonest_labels_data,on='track_id',how='inner')
	echonest_full_data = echonest_full_data.merge(echonest_data,on='track_id',how='inner')

	return echonest_full_data
	
#-------------------------------------------------------------------------------------------------------------------------------------------

def main():

	genres_file = '/home/ayush/FOML_Project/fma_metadata/genres.csv'
	tracks_file = '/home/ayush/FOML_Project/fma_metadata/tracks.csv'
	echonest_file = '/home/ayush/FOML_Project/fma_metadata/echonest.csv'
	librosa_file = '/home/ayush/FOML_Project/fma_metadata/features.csv'


	dict_genre_title_to_genre_id = map_genre_title_to_genre_id(genres_file)
	dict_track_id_to_parent_genre_title = map_track_id_to_parent_genre_title(tracks_file)
	process_echonest_tracks(echonest_file,dict_genre_title_to_genre_id,dict_track_id_to_parent_genre_title)


	#Call the function which merges features from echonest and librosa for echonest tracks
	echonest_labels = '/home/ayush/FOML_Project/echonest_labels.csv'
	echonest_full_data = merge_echonest_librosa_features(echonest_labels,echonest_file,librosa_file)

	print echonest_full_data.shape
	echonest_full_data.to_csv('echonest_full_data.csv',index=[0,1],sep=',',header=True)

	print echonest_full_data.head()


if __name__ == '__main__':
	main()