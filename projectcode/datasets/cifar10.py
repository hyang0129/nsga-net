import torchvision
from torchvision import transforms
import search.cifar10_search as my_cifar10
from loguru import logger
import numpy as np
import torch
import os


class CifarDataset(torch.utils.data.Dataset):
    def __init__(
        self, path="data.npz", transform: torchvision.transforms.Compose = None
    ):
        """
        Better performance cifar dataset, with all values loaded into memory. Reduces read pressure on disk.

        Args:
            path: save path of the zipped np arrays
            transform:
        """
        self.path = path

        if not os.path.exists(path):
            self._setup_dataset()

        f = np.load(path)
        self.x = f["x"]
        self.y = f["y"]
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def _setup_dataset(self, data_root: str = "../data"):

        logger.info("Downloading the CIFAR10 Dataset")

        train_transform = transforms.Compose([transforms.ToTensor()])

        train_data = my_cifar10.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )

        x = np.zeros((40000, 3, 32, 32)).astype(np.uint8)
        y = []

        for i in range(0, len(train_data)):
            ex = train_data.__getitem__(i)

            x[i] = (ex[0].numpy() * 255).astype(np.uint8)

            y.append(ex[1])

        y = np.array(y)
        x = x.transpose((0, 2, 3, 1))

        np.savez(self.path, x=x, y=y)

        logger.info("Successfully Downloaded CIFAR10 Dataset")
