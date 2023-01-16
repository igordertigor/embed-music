import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl


class PrintShape(nn.Module):
    def __init__(self, label=''):
        super().__init__()
        self.label = label

    def forward(self, x):
        print(self.label, x.size())
        return x


class SoundnetGenreClassifier(pl.LightningModule):
    learning_rate: float
    best_accuracy: float
    best_loss: float
    schedule_lr: bool

    def __init__(self, learning_rate: float = 1e-4, schedule_lr: bool = False):
        super().__init__()

        self.learning_rate = learning_rate
        self.best_loss = 0.
        self.best_accuracy = float('inf')
        self.schedule_lr = schedule_lr

        self.soundnet = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2),
            nn.Conv1d(16, 32, kernel_size=32, stride=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2),
            nn.Conv1d(32, 64, kernel_size=16, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.LogSoftmax(dim=-1),
        )

    def training_step(self, batch, batch_idx):
        genres, audio = batch
        h = self.soundnet(audio)
        y = self.decoder(h.view(h.size(0), -1))
        loss = nn.functional.cross_entropy(y, genres)
        return loss

    def validation_step(self, batch, batch_idx):
        genres, audio = batch
        h = self.soundnet(audio)
        y = self.decoder(h.view(h.size(0), -1))
        loss = nn.functional.cross_entropy(y, genres)
        self.best_loss = min(self.best_loss, loss.item())
        self.log('val_loss', loss)
        self.log('best_loss', self.best_loss)

        pred = y.max(1)
        accuracy = (pred.indices == genres).float().mean()
        self.best_accuracy = max(self.best_accuracy, accuracy)
        self.log('accuracy', accuracy)
        self.log('best_accuracy', self.best_accuracy)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        if self.schedule_lr:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': 'val_loss',
                    'frequency': 1,
                },
            }
        else:
            return optimizer


if __name__ == '__main__':
    model = SoundnetGenreClassifier()
    d = torch.randn(4, 1, 30000)
    h = model.soundnet(d)
    print(model.decoder(h.view(h.size(0), -1)))
