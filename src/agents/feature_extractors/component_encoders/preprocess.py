from typing import *

import torch


def preprocess_graph_representation(
    node_feature: torch.Tensor,
    edge_index: torch.Tensor,
    edge_feature: torch.Tensor,
    lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keys, values of observation
    node_feature: (batch, num_nodes, node_dim)
    edge_index: (batch, 2, num_edges)
    edge_feature: (batch, num_edges, edge_dim)
    lens: (batch, 1, 2)
    """
    if not isinstance(node_feature, torch.Tensor):
        node_feature = torch.Tensor(node_feature)
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.Tensor(edge_index)
    if not isinstance(edge_feature, torch.Tensor):
        edge_feature = torch.Tensor(edge_feature)
    if not isinstance(lens, torch.Tensor):
        lens = torch.Tensor(lens)

    lens = lens.long()
    n_nodes = []
    n_edges = []
    for i in range(len(lens)):
        n_nodes.append(lens[i][0])
        n_edges.append(lens[i][1])

    node_fea_batch = torch.cat(
        [node_feature[i][: n_nodes[i]] for i in range(len(node_feature))]
    )
    nNodes_cumsum = torch.cat(
        [torch.Tensor([0]), torch.cumsum(torch.Tensor(n_nodes).int(), dim=0)[:-1]]
    ).to(torch.int64)
    edge_index_batch = torch.cat(
        [
            edge_index[i][:, : n_edges[i]] + nNodes_cumsum[i]
            for i in range(len(edge_index))
        ],
        dim=1,
    ).long()
    edge_fea_batch = torch.cat(
        [edge_feature[i][: n_edges[i]] for i in range(len(edge_feature))]
    )
    batch_index = torch.cat(
        [torch.Tensor([i] * n_nodes[i]) for i in range(len(n_nodes))]
    ).to(torch.int64)

    return node_fea_batch, edge_index_batch, edge_fea_batch, batch_index

