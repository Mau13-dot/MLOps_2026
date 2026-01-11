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
    - accepts filter_data kwarg
    - clips values to [0,255] BEFORE uint8 conversion
    - when filter_data=True, removes black/white outliers using mean intensity
    - lazy H5 loading
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
        filter_data: bool = False,
        mean_low: float = 1.0,
        mean_high: float = 254.0,
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

        self._x_h5: Optional[h5py.File] = None
        self._y_h5: Optional[h5py.File] = None
        self.x_data = None
        self.y_data = None
        self._n: Optional[int] = None

        self._indices: Optional[np.ndarray] = None

    def _ensure_open(self) -> None:
        if self._x_h5 is None or self._y_h5 is None:
            self._x_h5 = h5py.File(str(self.x_path), "r")
            self._y_h5 = h5py.File(str(self.y_path), "r")
            self.x_data = self._x_h5["x"]
            self.y_data = self._y_h5["y"]
            self._n = int(len(self.x_data))

    def _ensure_indices(self) -> None:
        self._ensure_open()
        if self._indices is not None:
            return

        assert self._n is not None

        if not self.filter_data:
            self._indices = np.arange(self._n, dtype=np.int64)
            return

        keep_chunks = []
        for start in range(0, self._n, self.chunk_size):
            end = min(start + self.chunk_size, self._n)

            x = np.asarray(self.x_data[start:end])  

            x = np.clip(x, 0, 255).astype(np.uint8)

            means = x.mean(axis=(1, 2, 3)) 
            mask = (means > self.mean_low) & (means < self.mean_high)

            idxs = np.nonzero(mask)[0] + start
            keep_chunks.append(idxs.astype(np.int64))

        self._indices = (
            np.concatenate(keep_chunks) if keep_chunks else np.array([], dtype=np.int64)
        )
    
    def indices(self) -> np.ndarray:
        """Public access to (filtered) indices — required by the test suite."""
        self._ensure_indices()
        assert self._indices is not None
        return self._indices

    def __len__(self) -> int:
        self._ensure_indices()
        assert self._indices is not None
        return int(len(self._indices))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_indices()
        assert self._indices is not None
        self._ensure_open()
        assert self.x_data is not None and self.y_data is not None

        real_idx = int(self._indices[idx])

        image = np.asarray(self.x_data[real_idx])
        label = int(np.asarray(self.y_data[real_idx]).squeeze())

        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def close(self) -> None:
        if self._x_h5 is not None:
            self._x_h5.close()
        if self._y_h5 is not None:
            self._y_h5.close()

        self._x_h5 = None
        self._y_h5 = None
        self.x_data = None
        self.y_data = None
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
        state["x_data"] = None
        state["y_data"] = None
        return state
