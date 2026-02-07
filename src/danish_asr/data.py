"""Data loading and preprocessing.

Implement your dataset and LightningDataModule here.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    """Base dataset class. Implement __getitem__ and __len__ for your data."""

    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        # TODO: Load your data here

    def __len__(self) -> int:
        raise NotImplementedError("Implement __len__ for your dataset")

    def __getitem__(self, idx: int):
        raise NotImplementedError("Implement __getitem__ for your dataset")


class BaseDataModule(pl.LightningDataModule):
    """Lightning DataModule. Configure your data pipeline here."""

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data.get("data_dir", "data"))
        self.batch_size = cfg.data.get("batch_size", 32)
        self.num_workers = cfg.data.get("num_workers", 4)

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage (fit/test/predict)."""
        if stage == "fit" or stage is None:
            self.train_dataset = BaseDataset(self.data_dir, split="train")
            self.val_dataset = BaseDataset(self.data_dir, split="val")
        if stage == "test" or stage is None:
            self.test_dataset = BaseDataset(self.data_dir, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
