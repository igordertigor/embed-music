import os
import pandas as pd

columns_to_keep = ['track_id', 'album_id', 'track_genres']
tracks = pd.read_csv('data/fma_metadata/raw_tracks.csv')[columns_to_keep]


def has_audio(track_id: int) -> bool:
    track_str = str(track_id).zfill(6)
    fname = f'data/fma_small/{track_str[:3]}/{track_str}.mp3'
    return os.path.exists(fname)


tracks_small = tracks[tracks['track_id'].apply(has_audio)]
tracks_small.to_csv('out/tracks_small.csv')
