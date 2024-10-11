from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.bn_input = nn.BatchNorm1d(input_size)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes) - 2):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.dropout = dropout
        self.output_size = output_size

        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        x = self.bn_input(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout)

        x = F.normalize(x)
        return x


class MLPOneLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, output_size))

        self.dropout = dropout
        self.output_size = output_size
        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout)

        x = F.normalize(x)
        return x
