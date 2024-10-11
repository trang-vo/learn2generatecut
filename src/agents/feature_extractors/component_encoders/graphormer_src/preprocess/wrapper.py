# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from time import time
import _pickle

import torch
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def gen_edge_input(max_dist, path, edge_feat):
    nrows, ncols = path.shape
    assert nrows == ncols

    path_copy = path.copy()
    edge_feature_copy = edge_feat.copy()
    edge_fea_all = -1 * np.ones([nrows, ncols, max_dist, edge_feat.shape[-1]])

    for i in range(nrows):
        for j in range(ncols):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + algos.get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feature_copy[path[k], path[k + 1], :]

    return edge_fea_all


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    # x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)])
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr

    edge_input = attn_edge_type.unsqueeze(2)
    spatial_pos = np.full([N, N], -1, dtype=np.int64)
    spatial_pos[edge_index[0, :], edge_index[1, :]] = 1
    spatial_pos = torch.from_numpy(spatial_pos).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).float() if not isinstance(edge_input, torch.Tensor) else edge_input

    return item

