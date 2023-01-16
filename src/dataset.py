from typing import Tuple
import os
from enum import Enum
import pandas as pd
from torch.utils.data import Dataset
import torch
import torchaudio
from torchaudio.transforms import Resample


class Genre(Enum):
    ELECTRONIC = 15
    EXPERIMENTAL = 38
    FOLK = 17
    HIP_HOP = 21
    INTERNATIONAL = 2
    INSTRUMENTAL = 1235
    POP = 10
    ROCK = 12

N_TRAINING_EXAMPLES_PER_TRACK: int = 20


class FMA(Dataset):
    index_csv: str
    datadir: str

    def __init__(self, index_csv: str, datadir: str = 'data/raw/fma_small'):
        self.index_csv = index_csv
        self.datadir = datadir

        self._index = pd.read_csv(index_csv)

        self._genre_index = {g.value: i for i, g in enumerate(Genre)}
        # Might needs more controls here
        self.resampler = Resample(
            44100,
            8000,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return N_TRAINING_EXAMPLES_PER_TRACK*len(self._index)

    def __getitem__(self, i: int) -> Tuple[int, torch.FloatTensor]:
        idx = i // N_TRAINING_EXAMPLES_PER_TRACK
        offset = i % N_TRAINING_EXAMPLES_PER_TRACK
        record = self._index.iloc[idx]
        genre_id = self._genre_index[int(record['genre_top'])]
        track = str(int(record['track_id'])).zfill(6)
        filename = os.path.join(self.datadir, track[:3], f'{track}.mp3')

        try:
            audio, sampling_rate = torchaudio.load(
                filename,
                frame_offset=44100*offset,
                num_frames=int(44100*3.75),
            )
        except Exception:
            print('Failed to decode file', filename)
            raise
        return genre_id, self.resampler(audio).mean(dim=0, keepdims=True)


if __name__ == '__main__':
    data = FMA('data/processed/testing_set.csv')

    i, t = data[3]
    print(t.size())
    print(t.size(1)/8000)
