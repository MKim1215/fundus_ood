import torch
import math
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler, DistributedSampler
from .datasets.ood_dataset import Ood_Dataset


class Ood_DataLoader(DataLoader):
    def __init__(self):
        self.train_transform = T.Compose([
            T.ToTensor(),
        ])

        self.valid_transform = T.Compose([
            T.ToTensor(),
        ])
        self.setup()

    def setup(self):
        self.trainset = Ood_Dataset(self.data_dir, transform=self.train_transform)
        self.valset = Ood_Dataset(self.data_dir, transform=self.valid_transform)
        self.testset = Ood_Dataset(self.data_dir, transform=self.valid_transform)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size)

    def valid_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size)

    