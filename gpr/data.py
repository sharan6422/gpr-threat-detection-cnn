"""Patch dataset and a scan-level train/val/test split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, train: bool = False):
        self.files = df["filename"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.img_dir = Path(img_dir)
        self.train = train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = np.asarray(Image.open(self.img_dir / self.files[i]).convert("L"), dtype=np.float32) / 255.0
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()  # horizontal flip: a B-scan target looks the same mirrored along the scan
        x = torch.from_numpy((img - 0.5) / 0.5).unsqueeze(0)
        return x, self.labels[i]


def split_by_scan(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split so that all patches from one B-scan land in the same set.

    Shifted threat patches and neighbouring background windows from the same scan
    are near-duplicates. A random patch-level split would put copies of the same
    target in train and test and inflate the test score.
    """
    groups = df["source"]
    outer = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=groups))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["source"]))
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test
