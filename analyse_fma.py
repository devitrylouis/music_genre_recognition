# Analysis of FMA Dataset
# Not in Current Use


import pandas as pd 
import numpy as np

#--------------------------------------------------------------------------------------------------------------
# Raw Tracks
raw_tracks_files = '/home/ayush/FOML_Project/fma_metadata/raw_tracks.csv'
raw_tracks_data = pd.read_csv(raw_tracks_files,index_col=0)
print raw_tracks_data['track_genres']


#--------------------------------------------------------------------------------------------------------------
# Raw Genres
raw_genres_file = '/home/ayush/FOML_Project/fma_metadata/raw_genres.csv'
raw_genres_data = pd.read_csv(raw_genres_file,index_col=0)
print raw_genres_data.columns
print raw_genres_data['genre_parent_id']



#-------------------------------------------------------------------------------------------------------------
# Raw Artist Data
raw_artist_file = '/home/ayush/FOML_Project/fma_metadata/raw_artists.csv'
raw_artist_data = pd.read_csv(raw_artist_file,index_col=0)
print raw_artist_data.columns


#---------------------------------------------------------------------------------------------------------
# Raw Albums Data
raw_album_file = '/home/ayush/FOML_Project/fma_metadata/raw_albums.csv'
raw_album_data = pd.read_csv(raw_album_file,index_col=0)
print raw_album_data.columns

#----------------------------------------------------------------------------------------------------------
#Raw Echonest Data
raw_echonest_file = '/home/ayush/FOML_Project/fma_metadata/raw_echonest.csv'
raw_echonest_data = pd.read_csv(raw_echonest_file,index_col=0)
#print raw_echonest_data.columns


#--------------------------------------------------------------------------------------------------------------
# Tracks : per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
tracks_file = '/home/ayush/FOML_Project/fma_metadata/tracks.csv'
tracks_data = pd.read_csv(tracks_file,delimiter=',',index_col=0,header=[0,1])
#print tracks_data.columns
#print 'tracks data size' ,tracks_data.shape


#---------------------------------------------------------------------------------------------------------------
# Genres Data : all 163 genre IDs with their name and parent (used to infer the genre hierarchy and top-level genres).
genres_file = '/home/ayush/FOML_Project/fma_metadata/genres.csv'
genres_data = pd.read_csv(genres_file,delimiter=',',index_col=0,header=[0,1])
#print genres_data.columns
#print 'genres_data size', genres_data.shape


#---------------------------------------------------------------------------------------------------------------
# Features Data : common features extracted with librosa
features_file = '/home/ayush/FOML_Project/fma_metadata/features.csv'
features_data = pd.read_csv(features_file,delimiter=',',index_col=0,header=[0,1,2])
#print features_data.columns
#print 'features_data size', features_data.shape


#---------------------------------------------------------------------------------------------------------------
# Echonest Data : audio features provided by Echonest (now Spotify) for a subset of 13,129 tracks
echonest_file = '/home/ayush/FOML_Project/fma_metadata/echonest.csv'
echonest_data = pd.read_csv(echonest_file,delimiter=',',index_col=0,header=[0,1,2])
#print echonest_data.columns
#print 'echonest data size' ,  echonest_data.shape #13129x249