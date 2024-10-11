import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GINEConv

from .preprocess import preprocess_graph_representation


class GINEGraphExtractor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pre_mp = nn.Linear(node_dim, hidden_size)
        self.pre_edge_feature = nn.Linear(edge_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(self.num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.convs.append(GINEConv(layer))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.post_mp = nn.Linear(hidden_size, hidden_size)

        self.output_size = hidden_size
        self.device = device

    def forward(
        self, items
    ):
        x = items.node_feature
        edge_index = items.edge_index
        edge_feature = items.edge_feature
        lens = items.lens
        x, edge_index, edge_feature, batch = preprocess_graph_representation(
            x, edge_index, edge_feature, lens
        )
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_feature = edge_feature.to(self.device)
        batch = batch.to(self.device)

        x = self.pre_mp(x)
        edge_feature = self.pre_edge_feature(edge_feature)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_feature)
            x = self.bns[i](x)
            # x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.normalize(x)

        return x