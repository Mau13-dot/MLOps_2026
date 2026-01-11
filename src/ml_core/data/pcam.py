from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.

    Test expectations:
    - Accepts filter_data kwarg
    - Clips values to [0,255] BEFORE casting to uint8
    - Mean-based filtering drops black/white outliers (mean == 0 and mean == 255)
    - Exposes filtered indices as `ds.indices`
    - Lazy H5 loading (safe with DataLoader workers)
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
        filter_data: bool = False,
        mean_low: float = 0.0,     # keep if mean > 0
        mean_high: float = 255.0,  # keep if mean < 255
        chunk_size: int = 1024,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        self.filter_data = bool(filter_data)
        self.mean_low = float(mean_low)
        self.mean_high = float(mean_high)
        self.chunk_size = int(chunk_size)

        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                f"PCAM files not found at {self.x_path} or {self.y_path}"
            )

        # Lazy handles
        self._x_h5: Optional[h5py.File] = None
        self._y_h5: Optional[h5py.File] = None
        self._x_ds = None
        self._y_ds = None
        self._n: Optional[int] = None

        # Computed lazily
        self._indices: Optional[np.ndarray] = None

    def _ensure_open(self) -> None:
        if self._x_h5 is None or self._y_h5 is None:
            self._x_h5 = h5py.File(str(self.x_path), "r")
            self._y_h5 = h5py.File(str(self.y_path), "r")
            self._x_ds = self._x_h5["x"]
            self._y_ds = self._y_h5["y"]
            self._n = int(len(self._x_ds))

    def _ensure_indices(self) -> None:
        self._ensure_open()
        if self._indices is not None:
            return

        assert self._n is not None
        assert self._x_ds is not None

        if not self.filter_data:
            self._indices = np.arange(self._n, dtype=np.int64)
            return

        keep_parts = []
        for start in range(0, self._n, self.chunk_size):
            end = min(start + self.chunk_size, self._n)
            x = np.asarray(self._x_ds[start:end])

            
            x = np.clip(x, 0, 255).astype(np.uint8)

            means = x.mean(axis=(1, 2, 3))
            
            mask = (means > self.mean_low) & (means < self.mean_high)

            idxs = np.nonzero(mask)[0] + start
            keep_parts.append(idxs.astype(np.int64))

        self._indices = (
            np.concatenate(keep_parts) if keep_parts else np.array([], dtype=np.int64)
        )

    @property
    def indices(self) -> np.ndarray:
        """Public access to filtered indices (required by tests)."""
        self._ensure_indices()
        assert self._indices is not None
        return self._indices

    def __len__(self) -> int:
        self._ensure_indices()
        assert self._indices is not None
        return int(len(self._indices))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_indices()
        self._ensure_open()
        assert self._indices is not None
        assert self._x_ds is not None and self._y_ds is not None

        real_idx = int(self._indices[idx])

        image = np.asarray(self._x_ds[real_idx])
        label = int(np.asarray(self._y_ds[real_idx]).squeeze())

        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform is not None:
            image_t = self.transform(image)
        else:
            
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_t = torch.tensor(label, dtype=torch.long)
        return image_t, label_t

    def close(self) -> None:
        if self._x_h5 is not None:
            self._x_h5.close()
        if self._y_h5 is not None:
            self._y_h5.close()
        self._x_h5 = None
        self._y_h5 = None
        self._x_ds = None
        self._y_ds = None
        self._n = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        
        state = self.__dict__.copy()
        state["_x_h5"] = None
        state["_y_h5"] = None
        state["_x_ds"] = None
        state["_y_ds"] = None
        return state
