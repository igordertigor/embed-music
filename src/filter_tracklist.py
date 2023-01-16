import os
import pandas as pd
import torchaudio

columns_to_keep = ['track_id', 'album_id', 'track_genres']
tracks = pd.read_csv('data/raw/fma_metadata/raw_tracks.csv')[columns_to_keep]


def has_audio(track_id: int) -> bool:
    track_str = str(track_id).zfill(6)
    fname = f'data/raw/fma_small/{track_str[:3]}/{track_str}.mp3'
    if os.path.exists(fname) and os.path.getsize(fname) > 4000:
        try:
            audio, sampling_rate = torchaudio.load(fname)
        except Exception:
            return False
        return sampling_rate == 44100
    else:
        # print(f'File {fname} does not exist')
        return False


tracks_small = tracks[tracks['track_id'].apply(has_audio)]
tracks_small.to_csv('data/processed/tracks_small.csv', index=False)
