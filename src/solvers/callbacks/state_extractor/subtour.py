from typing import *

import networkx as nx
import numpy as np

from solvers.callbacks.state_extractor.base import BaseStateExtractor
from problems.tsp import TSPProblem
from solvers.callbacks.separators.subtour import SubtourSeparator
from solvers.cplex_api import UserCutCallback
from constant import TOLERANCE
from utils import nodes2edge, PseudoTorchData
from config import EnvConfig


class SubtourStateExtractor(BaseStateExtractor):
    def initialize_original_graph(
            self, problem: TSPProblem, var2idx: Dict[Tuple[int, int], int], k=10
    ) -> None:
        """Create a graph representation involving (node feature, edge feature, edge index) for the original graph.
        If the graph is completed, use k-nearest neighbors graph."""
        self.problem = problem

        k_nearest_neighbors: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        for node in problem.graph.nodes:
            k_nearest_neighbors[node] = sorted(
                problem.graph.adj[node].items(), key=lambda e: e[1]["weight"]
            )[:k]

        max_weight: int = max(
            [problem.graph.edges[edge]["weight"] for edge in problem.graph.edges]
        )
        knn_graph = nx.Graph()
        for node in k_nearest_neighbors:
            for neighbor, node_dict in k_nearest_neighbors[node]:
                knn_graph.add_edge(
                    node, neighbor, weight=node_dict["weight"] / max_weight
                )

        self.ori_node_feature = []
        self.ori_edge_index = [[], []]
        self.ori_edge_weight = []
        self.ori_edge_index_list = []

        for node in knn_graph:
            self.ori_node_feature.append([len(knn_graph.adj[node])])
            for neighbor in knn_graph.adj[node]:
                self.ori_edge_index[0].append(node)
                self.ori_edge_index[1].append(neighbor)
                self.ori_edge_weight.append(knn_graph[node][neighbor]["weight"])
                self.ori_edge_index_list.append(var2idx[nodes2edge(node, neighbor)])

        self.ori_node_feature = np.asarray(self.ori_node_feature)
        self.ori_edge_index = np.asarray(self.ori_edge_index)
        self.ori_lens = np.array(
            [self.ori_node_feature.shape[0], self.ori_edge_index.shape[1]]
        )

        if self.padding:
            self.max_nb_edges = len(problem.graph.nodes) * k * 2
            self.ori_edge_index = np.concatenate((self.ori_edge_index, np.zeros((2, self.max_nb_edges - self.ori_lens[1]))), axis=1)
            self.ori_edge_index.astype(np.uint16)

    def compress_support_graph(self, support_graph: nx.Graph) -> nx.Graph:
        g = nx.Graph()
        for node in support_graph.nodes:
            if len(support_graph.adj[node]) == 2:
                for neighbor in support_graph.adj[node]:
                    g.add_edge(node, neighbor)

        compress_node_label = {}
        infor_component = {}
        components = list(nx.connected_components(g))
        for idx, component in enumerate(components):
            if len(component) > 4:
                endpoints = []
                for node in component:
                    if len(g.adj[node]) == 1:
                        compress_node_label[node] = (idx, 0)
                        for neighbor in g.adj[node]:
                            compress_node_label[neighbor] = (idx, 1)
                            endpoints.append(neighbor)

                if len(endpoints) == 2:
                    infor_component[idx] = endpoints
                else:
                    component = list(component)
                    tmp_node = component[0]
                    compress_node_label[tmp_node] = (idx, 0)
                    for neighbor in g.adj[tmp_node]:
                        compress_node_label[neighbor] = (idx, 1)
                    infor_component[idx] = list(g.adj[tmp_node])
            else:
                for node in component:
                    compress_node_label[node] = (idx, 0)

        output = nx.Graph()
        for node in support_graph.nodes:
            if len(support_graph.adj[node]) > 2:
                for neighbor in support_graph.adj[node]:
                    output.add_edge(
                        node,
                        neighbor,
                        weight=support_graph.adj[node][neighbor]["weight"],
                    )
            else:
                if node in compress_node_label:
                    idx, label = compress_node_label[node]
                    if label == 1:
                        for neighbor in support_graph.adj[node]:
                            if neighbor in compress_node_label:
                                output.add_edge(
                                    node,
                                    neighbor,
                                    weight=support_graph[node][neighbor]["weight"],
                                )

                        endpoints = infor_component[idx]
                        output.add_edge(endpoints[0], endpoints[1], weight=1)
                    else:
                        for neighbor in support_graph.adj[node]:
                            output.add_edge(
                                node,
                                neighbor,
                                weight=support_graph.adj[node][neighbor]["weight"],
                            )

        new_label = {}
        for node in output.nodes:
            new_label[node] = len(new_label)

        return nx.relabel_nodes(output, new_label)

    def get_original_graph_representation(
            self,
            solution: np.array,
            lb: np.array,
            ub: np.array,
            *args, **kwargs
    ) -> Dict[str, np.array]:
        ori_edge_feature = []
        for idx, edge_idx in enumerate(self.ori_edge_index_list):
            ori_edge_feature.append(
                [
                    self.ori_edge_weight[idx],
                    solution[edge_idx],
                    lb[edge_idx],
                    ub[edge_idx],
                ]
            )

        ori_edge_feature = np.asarray(ori_edge_feature)

        if self.padding:
            ori_edge_feature = np.concatenate(
                (ori_edge_feature, np.zeros((self.max_nb_edges - self.ori_lens[1], ori_edge_feature.shape[1]))), 
                axis=0
            )

        return dict(
            node_feature=self.ori_node_feature,
            edge_index=self.ori_edge_index,
            edge_feature=ori_edge_feature,
            lens=self.ori_lens,
        )

    def get_solution_representation(self, solution_graph: nx.Graph, *args, **kwargs) -> PseudoTorchData:
        g = self.compress_support_graph(solution_graph)
        return super(SubtourStateExtractor, self).get_solution_representation(g)

    def get_cut_detector_solution_representation(self, solution_graph: nx.Graph) -> Dict[str, np.ndarray]:
        g = self.compress_support_graph(solution_graph)
        item = super().get_solution_representation(g)
        return dict(
            sup_node_feature=item.x,
            sup_edge_index=np.asarray(item.edge_index),
            sup_edge_feature=np.asarray(item.edge_attr),
            sup_lens=np.array([item.x.shape[0], item.edge_index.shape[1]]),
        )


class GraphormerSubtourStateExtractor(SubtourStateExtractor):
    def get_solution_representation(self, solution_graph: nx.Graph, *args, **kwargs) -> PseudoTorchData:
        g = self.compress_support_graph(solution_graph)
        item = super().get_solution_representation(g, *args, **kwargs)

        node_feature, edge_index, edge_feature = item.x, item.edge_index, item.edge_attr
        N = node_feature.shape[0]
        edge_index = np.asarray(edge_index)
        edge_feature = np.asarray(edge_feature)

        adj = nx.adjacency_matrix(g).todense()
        attn_edge_type = np.zeros((adj.shape[0], adj.shape[1], edge_feature.shape[1]))
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_feature

        spatial_pos = np.full([N, N], -1, dtype=np.int16)
        spatial_pos[edge_index[0, :], edge_index[1, :]] = 1
        attn_bias = np.zeros([N + 1, N + 1])
        edge_input = np.expand_dims(attn_edge_type, axis=2)

        item.attn_bias = attn_bias
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = spatial_pos
        item.in_degree = np.sum(np.asarray(adj), axis=1)
        item.out_degree = item.in_degree
        item.edge_input = edge_input

        return item
