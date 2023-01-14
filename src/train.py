from typing import Optional
from pydantic import BaseModel
from yaml import safe_load
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dvclive.lightning import DVCLiveLogger
import time

# import mlem

from src.model import SoundnetGenreClassifier
from src.dataset import FMA


class Config(BaseModel):
    batch_size: int = 16
    num_workers: int = 3
    max_epochs: int = -1
    learning_rate: float = 1e-4


if __name__ == '__main__':
    with open('params.yaml') as f:
        config = Config(**safe_load(f))

    model = SoundnetGenreClassifier(config.learning_rate)
    dl_train = DataLoader(
        FMA('data/final/training_set.csv'),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        FMA('data/final/validation_set.csv'),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    checkpoints = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='experiments/training/checkpoints',
        save_last=True,
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=config.max_epochs,
        callbacks=[checkpoints],
        logger=DVCLiveLogger(
            'experiments/training',
            dir='experiments/training',
        )
    )
    t0 = time.time()
    trainer.fit(
        model=model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_val,
    )
    print('Total training time', time.time() - t0)

    # print(model)
    # torch.save(model, 'models/model.pt')

    # logger, = [
    #     log
    #     for log in trainer.loggers
    #     if isinstance(log, pl.loggers.CSVLogger)
    # ]

    # best_record = {'val_loss': 1e10}
    # for record in logger.experiment.metrics:
    #     if record['val_loss'] < best_record['val_loss']:
    #         best_record = record
    # with open('data/final/metrics.json', 'w') as f:
    #     json.dump(best_record, f)
    # print(best_record)

    # mlem.api.save(model, 'sound-genre-classifier')
