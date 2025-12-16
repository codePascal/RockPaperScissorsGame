#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from pytorch_lightning.callbacks import ModelCheckpoint

# Hyper-parameters
RANDOM_STATE = 42
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
LEARNING_RATE = 1e-3
EPOCHS = 10

# Dataset root
DATA_DIR = Path.home().joinpath('.kaggle', 'rock-paper-scissors')

# Model state dict
MODEL_PT = Path(__file__).parent.parent.joinpath('out', 'rps_mobilenetv2.pt')

# Transform to be applied on input images.
TRANSFORM_ = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Available classes: sorted as depicted in ImageFolder (alphabetical)
CLASSES = {
    0: 'paper',
    1: 'rock',
    2: 'scissors'
}


class RPSDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: Path,
            transform: transforms.Compose,
            batch_size: int = BATCH_SIZE,
            train_split: float = TRAIN_SPLIT,
            num_workers: int = 2
    ):
        super().__init__()

        self.transform = transform
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers

        self.train_dir = data_dir.joinpath('rps', 'rps')
        self.test_dir = data_dir.joinpath('rps-test-set', 'rps-test-set')

        self.train_dl = Type[DataLoader]
        self.val_dl = Type[DataLoader]
        self.test_dl = Type[DataLoader]

    def setup(self, stage: str = None):
        rps_train = datasets.ImageFolder(
            self.train_dir,
            transform=self.transform
        )
        rps_test = datasets.ImageFolder(
            self.test_dir,
            transform=self.transform
        )

        n_train_total = len(rps_train)
        n_train = int(self.train_split * n_train_total)
        n_val = n_train_total - n_train
        train_set, val_set = random_split(
            rps_train,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(RANDOM_STATE)
        )
        test_set = rps_test

        self.train_dl = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )
        self.val_dl = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        self.test_dl = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


class RPSLiteModel(pl.LightningModule):

    def __init__(self, lr: float = 1e-3):
        super().__init__()

        self.save_hyperparameters('lr')

        n_classes = len(CLASSES.keys())

        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, n_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_train = Accuracy(task='multiclass', num_classes=n_classes)
        self.acc_val = Accuracy(task='multiclass', num_classes=n_classes)
        self.acc_test = Accuracy(task='multiclass', num_classes=n_classes)

    def _add_log(self, name, metric):
        self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc_train(logits, y)
        self._add_log('train/loss', loss)
        self._add_log('train/acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc_val(logits, y)
        self._add_log('val/loss', loss)
        self._add_log('val/acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc_test(logits, y)
        self._add_log('test/loss', loss)
        self._add_log('test/acc', acc)


def main():
    pl.seed_everything(RANDOM_STATE, workers=True)

    data_module = RPSDataModule(
        data_dir=DATA_DIR,
        transform=TRANSFORM_,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        num_workers=2
    )
    model = RPSLiteModel(lr=LEARNING_RATE)

    ckpt = ModelCheckpoint(
        filename='rps-{epoch:02d}-{val_acc:.4f}',
        monitor='val/acc',
        mode='max',
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        callbacks=ckpt
    )
    trainer.fit(model, data_module)
    print('Best checkpoint:', ckpt.best_model_path)

    trainer.test(model, data_module)
    MODEL_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), MODEL_PT)


if __name__ == '__main__':
    main()
