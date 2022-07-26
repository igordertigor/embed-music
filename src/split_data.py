import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('out/clean_tracks_small.csv')
data['track_id'] = data['track_id'].apply(lambda x: str(x).zfill(6))
data['album_id'] = data['album_id'].apply(int)
data['genre_top'] = data['genre_top'].apply(int)

tmp, test = train_test_split(data, test_size=80, stratify=data['genre_top'])
train, valid = train_test_split(tmp, test_size=80, stratify=tmp['genre_top'])

print(len(train), len(valid), len(test))

test.to_csv('out/testing_set.csv', index=False)
train.to_csv('out/training_set.csv', index=False)
valid.to_csv('out/validation_set.csv', index=False)
