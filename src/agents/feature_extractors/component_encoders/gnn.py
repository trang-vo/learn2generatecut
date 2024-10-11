from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from utils import PseudoTorchData
from .preprocess import preprocess_graph_representation


class GNNGraphExtractor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_size: int,
        num_layers: int,
        n_clusters: int = 2,
        dropout: float = 0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.pre_mp = nn.Linear(node_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphConv(hidden_size, hidden_size))
            self.bns.append(nn.BatchNorm1d(hidden_size))

        self.pool = nn.Linear(hidden_size, n_clusters)

        self.dropout = dropout
        self.output_size = hidden_size
        self.device = device

    def forward_(
        self, x: np.array, edge_index: np.array, edge_feature: np.array, lens: np.array
    ):
        x, edge_index, edge_feature, batch = preprocess_graph_representation(
            x, edge_index, edge_feature, lens
        )
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_feature = edge_feature.to(self.device)
        batch = batch.to(self.device)

        x = self.pre_mp(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_feature)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # mincut pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = self.pool(x)
        x, adj, mc_loss, o_loss = dense_mincut_pool(x, adj, s, mask)
        x = F.leaky_relu(x)

        # get graph embedding, size x = (batch_size, hidden_size)
        x = x.mean(dim=1)
        x = F.normalize(x)

        return x

    def forward(self, items: List[PseudoTorchData]):
        x = torch.Tensor(np.array([item.x for item in items]))
        edge_index = torch.tensor(np.array([item.edge_index for item in items])).long()
        edge_attr = torch.Tensor(np.array([item.edge_attr for item in items]))
        lens = torch.tensor(np.array([[item.x.shape[0], item.edge_index.shape[1]] for item in items])).long()

        return self.forward_(x, edge_index, edge_attr, lens)
