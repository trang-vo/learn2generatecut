from typing import *

import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import MLP


# Create an MLP model with embedding layers for categorical inputs
class EmbedMLP(nn.Module):
    def __init__(self, categorical_dims: Dict[str, int], continuous_dims: int, hidden_sizes: List[int],
                 output_size: int, embedding_dim: int = 1,
                 norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
                 device: Union[str, int, torch.device] = "cpu"):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_size = output_size

        self.continuous_linear = nn.Linear(continuous_dims, output_size).to(self.device)

        self.categorical_dims = categorical_dims
        self.embeddings = {
            feature: nn.Embedding(num_embeddings + 1, embedding_dim).to(self.device)
            for feature, num_embeddings in categorical_dims.items()
        }
        self.categorical_linear = nn.Linear(len(categorical_dims) * embedding_dim, output_size).to(self.device)

        input_dim = output_size * 2
        self.mlp = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_size, norm_layer=norm_layer,
                       device=device, flatten_input=False).to(self.device)

    def forward(self, x):
        categorical_features = x["categorical"]
        continuous_features = torch.Tensor(x["continuous"]).float().to(self.device)

        embedded_features = []
        for feature, values in categorical_features.items():
            values = np.where(values < self.categorical_dims[feature], values, self.categorical_dims[feature])
            values = torch.Tensor(values).long().to(self.device)
            embedded = self.embeddings[feature](values)
            embedded_features.append(embedded)

        categorical_embedded = self.categorical_linear(torch.cat(embedded_features, dim=1))
        continuous_embedded = self.continuous_linear(continuous_features)

        x = torch.cat([categorical_embedded, continuous_embedded], dim=1)
        x = self.mlp(x)

        return x
