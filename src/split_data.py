from typing import Tuple
import pandas as pd
from yaml import safe_load
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


class Config(BaseModel):
    test_set_size: float
    valid_set_size: float

    @property
    def train_set_size(self) -> int:
        return 1 - self.test_set_size - self.valid_set_size


def split_threefold(
    data: pd.DataFrame,
    test_set_size: float,
    valid_set_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmp, test = train_test_split(
        data,
        test_size=test_set_size,
        stratify=data['genre_top'],
        random_state=1,
    )

    train, valid = train_test_split(
        tmp,
        test_size=valid_set_size/(1-test_set_size),
        stratify=tmp['genre_top'],
        random_state=2,
    )
    return train, valid, test


if __name__ == '__main__':
    with open('params.yaml') as f:
        config = Config(**safe_load(f))

    data = pd.read_csv('data/processed/clean_tracks_small.csv')
    data['track_id'] = data['track_id'].apply(lambda x: str(x).zfill(6))
    data['album_id'] = data['album_id'].apply(int)
    data['genre_top'] = data['genre_top'].apply(int)

    train, valid, test = split_threefold(
        data,
        test_set_size=config.test_set_size,
        valid_set_size=config.valid_set_size,
    )

    print(len(train), len(valid), len(test))

    test.to_csv('data/final/testing_set.csv', index=False)
    train.to_csv('data/final/training_set.csv', index=False)
    valid.to_csv('data/final/validation_set.csv', index=False)
