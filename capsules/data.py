import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    """Synthetic Dataset Generator."""

    def __init__(self,
                 n: int = 128,
                 size: tuple = (32, 32, 32),
                 n_classes: int = 3):
        self.n = n

        self.X = torch.randn(n, 1, *size)
        self.y = torch.randint(0, 2, (n, n_classes, *size))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]


class SyntheticDataModule(LightningDataModule):
    """Synthetic DataModule used for training"""

    def __init__(self,
                 n_train: int = 128,
                 n_val: int = 16,
                 n_test: int = 16,
                 size: tuple = (32, 32, 32),
                 n_classes: int = 3,
                 batch_size: int = 1):
        super().__init__()
        self.batch_size = batch_size

        self.train = SyntheticDataset(n=n_train, size=size, n_classes=n_classes)
        self.val = SyntheticDataset(n=n_val, size=size, n_classes=n_classes)
        self.test = SyntheticDataset(n=n_test, size=size, n_classes=n_classes)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
