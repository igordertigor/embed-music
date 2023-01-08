import pandas as pd
import ast

tracks = pd.read_csv('data/processed/tracks_small.csv')
genres = pd.read_csv('data/raw/fma_metadata/genres.csv')

genre_lookup = {}
name_lookup = {}
for i, g in genres.iterrows():
    genre_lookup[g['genre_id']] = g['top_level']
    if g['genre_id'] == g['top_level']:
        name_lookup[g['top_level']] = g['title']

for i, t in tracks.iterrows():
    g = ast.literal_eval(t['track_genres'])
    gg = set()
    for g_ in g:
        genre_id = int(g_['genre_id'])
        if genre_id in genre_lookup:
            top_level = genre_lookup[genre_id]
            gg.add(top_level)
    if len(gg) > 1:
        raise ValueError
    if len(gg) == 0:
        raise ValueError
    genre_top = int(list(gg)[0])
    tracks.loc[i, 'genre_top'] = genre_top
    tracks.loc[i, 'genre_title'] = name_lookup[genre_top]

tracks['track_id'] = tracks['track_id'].apply(lambda x: str(x).zfill(6))
tracks['album_id'] = tracks['album_id'].apply(int)
tracks['genre_top'] = tracks['genre_top'].apply(int)

tracks[['track_id', 'album_id', 'genre_top', 'genre_title']].to_csv(
    'data/processed/clean_tracks_small.csv',
    index=False,
)
