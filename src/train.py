from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.model import SoundnetGenreClassifier
from src.dataset import FMA


model = SoundnetGenreClassifier()
dataset = FMA('out/training_set.csv')
dl = DataLoader(dataset, batch_size=16, num_workers=3)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=dl)
