from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        # TODO: Initialize dataset
        if not self . x_path . exists () or not self . y_path . exists ():
            raise FileNotFoundError (
            f" PCAM files not found at { self . x_path } or { self . y_path }
            "   
            )

        # 2. Open h5 files in read mode
        self . x_data = h5py . File ( self .x_path , "r")["x"]
        self . y_data = h5py . File ( self .y_path , "r")["y"]


    def __len__(self) -> int:
        # TODO: Return length of dataset
        def __len__ ( self ) -> int:
            return len( self . x_data )
        # The dataloader will know hence how many batches to create


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # TODO: Implement data retrieval
        # 1. Read data at idx
        image = self . x_data [idx]
        label = self . y_data [idx ][0]

        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        image = image . astype (np. uint8 )

        # 3. Apply transforms if they exist
        if self . transform :
            image = self . transform ( image )

        # 4. Return tensor image and label (as long)
        return image , torch . tensor (label , dtype = torch . long ). squeeze ()

        
        
