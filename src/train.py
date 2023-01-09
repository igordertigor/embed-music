from typing import Optional
from pydantic import BaseModel
from yaml import safe_load
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# import mlem

from src.model import SoundnetGenreClassifier
from src.dataset import FMA


class Config(BaseModel):
    batch_size: int = 16
    num_workers: int = 3
    limit_train_batches: Optional[int]
    max_epochs: Optional[int]
    learning_rate: float = 1e-4


if __name__ == '__main__':
    with open('params.yaml') as f:
        config = Config(**safe_load(f))

    model = SoundnetGenreClassifier(config.learning_rate)
    dataset = FMA('data/final/training_set.csv')
    dl = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    trainer = pl.Trainer(
        limit_train_batches=config.limit_train_batches,
        max_epochs=config.max_epochs,
    )
    trainer.fit(model=model, train_dataloaders=dl)

    torch.save(model, 'models/model.pt')

    # mlem.api.save(model, 'sound-genre-classifier')
