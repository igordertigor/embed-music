import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# import mlem

from src.model import SoundnetGenreClassifier
from src.dataset import FMA


model = SoundnetGenreClassifier()
dataset = FMA('data/final/training_set.csv')
dl = DataLoader(dataset, batch_size=16, num_workers=3)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=dl)

torch.save(model, 'models/model.pt')

# mlem.api.save(model, 'sound-genre-classifier')
