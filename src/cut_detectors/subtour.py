import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch.nn import Linear


class SubtourDetector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args, device="cpu"):
        super().__init__()
        self.dropout = args['dropout']
        self.pre_mp = nn.Linear(input_size, hidden_size)
        self.conv1 = GraphConv(hidden_size, hidden_size)
        self.bns1 = nn.BatchNorm1d(hidden_size)

        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.bns2 = nn.BatchNorm1d(hidden_size)

        self.pool1 = Linear(hidden_size, 2)
        self.post_mp = nn.Linear(2 * hidden_size, output_size)
        self.device = device

    def forward(self, data):
        x, edge_index, edge_feature, batch = data.node_feature, data.edge_index, data.edge_feature, data.batch
        x = self.pre_mp(x)

        # block 1
        x = self.conv1(x, edge_index, edge_feature)
        x = self.bns1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # block 2
        x = self.conv2(x, edge_index, edge_feature)
        x = self.bns2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # mincut pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch, edge_feature.view(-1))

        s = self.pool1(x)
        x, adj, mc_loss, o_loss = dense_mincut_pool(x, adj, s, mask)
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.post_mp(x)
        return x, mc_loss, o_loss

    def predict(self, x, edge_index, edge_feature, batch):
        # self.eval()
        x = torch.Tensor(x).to(self.device)
        edge_index = torch.Tensor(edge_index).long().to(self.device)
        edge_feature = torch.Tensor(edge_feature).to(self.device)
        batch = torch.Tensor(batch).long().to(self.device)
        with torch.no_grad():
            x = self.pre_mp(x)

            # block 1
            x = self.conv1(x, edge_index, edge_feature)
            x = self.bns1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # block 2
            x = self.conv2(x, edge_index, edge_feature)
            x = self.bns2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # mincut pooling
            x, mask = to_dense_batch(x, batch)
            adj = to_dense_adj(edge_index, batch, edge_feature.view(-1))

            s = self.pool1(x)
            x, adj, _, _ = dense_mincut_pool(x, adj, s, mask)
            x = F.relu(x)
            x = x.view(x.size()[0], -1)
            x = self.post_mp(x)
        return int(x.argmax())

    def get_graph_vec(self, x, edge_index, edge_feature, batch):
        x = torch.Tensor(x)
        edge_index = torch.Tensor(edge_index).long()
        edge_feature = torch.Tensor(edge_feature)
        batch = torch.Tensor(batch).long()
        x = self.pre_mp(x)

        # block 1
        x = self.conv1(x, edge_index, edge_feature)
        x = self.bns1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # block 2
        x = self.conv2(x, edge_index, edge_feature)
        x = self.bns2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # mincut pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch, edge_feature.view(-1))

        s = self.pool1(x)
        x, adj, mc_loss, o_loss = dense_mincut_pool(x, adj, s, mask)
        x = F.relu(x)
        x = x.view(x.size()[0], -1)

        return x
