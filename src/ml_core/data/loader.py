from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files. Uses WeightedRandomSampler for imbalance.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    base_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_transform = base_transform
    val_transform = base_transform

    train_x_path = base_path / "camelyonpatch_level_2_split_train_x.h5"
    train_y_path = base_path / "camelyonpatch_level_2_split_train_y.h5"
    val_x_path = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    val_y_path = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    # Optional: allow config to toggle filtering
    filter_data = bool(data_cfg.get("filter_data", False))

    train_ds = PCAMDataset(
        str(train_x_path),
        str(train_y_path),
        transform=train_transform,
        filter_data=filter_data,
    )
    val_ds = PCAMDataset(
        str(val_x_path),
        str(val_y_path),
        transform=val_transform,
        filter_data=False,  # usually don't filter validation
    )

    # Read labels once from the H5 file
    with h5py.File(str(train_y_path), "r") as f:
        y = np.asarray(f["y"][:]).squeeze().astype(np.int64)

    class_counts = np.bincount(y)
    if (class_counts == 0).any():
        raise ValueError(f"Missing class in training set. counts={class_counts}")

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,  
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    

    return train_loader, val_loader
    
