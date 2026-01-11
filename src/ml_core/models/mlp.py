from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        in_features = 1
        for d in input_shape:
            in_features *= int(d)

        layers: List[nn.Module] = []
        layers.append(nn.Flatten())

        prev = in_features
        for h in hidden_units:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=float(dropout_rate)))
            prev = int(h)

        layers.append(nn.Linear(prev, int(num_classes)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
