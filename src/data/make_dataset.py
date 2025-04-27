# -*- coding: utf-8 -*-
import multiprocessing
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CROPSIZE = 224
RESIZE = 256
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        base_path = os.path.join(config.paths.raw_data_path, config.data.name)
        self.train_dir = os.path.join(base_path, "train")
        self.val_dir = os.path.join(base_path, "val")

        self.batch_size = config.experiment.batch_size
        self.threads = 0  

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(CROPSIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(RESIZE),
                transforms.CenterCrop(CROPSIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ]
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train = datasets.ImageFolder(self.train_dir, self.train_transform)
            self.val = datasets.ImageFolder(self.val_dir, self.val_transform)
        if stage == "test":
            self.test = datasets.ImageFolder(self.val_dir, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.threads,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.threads,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.threads,
        )
