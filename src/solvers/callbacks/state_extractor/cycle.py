from typing import *
from typing import Dict

import numpy as np
import networkx as nx
from numpy import ndarray

from solvers.callbacks.state_extractor.base import BaseStateExtractor
from solvers.callbacks.separators.cycle import CycleSeparator
from solvers.cplex_api import UserCutCallback
from problems.maxcut import MaxcutProblem
from utils import nodes2edge, PseudoTorchData
from constant import TOLERANCE


class CycleStateExtractor(BaseStateExtractor):
    def initialize_original_graph(
            self, problem: MaxcutProblem, var2idx: Dict[Tuple[int, int], int], **kwargs
    ) -> None:
        self.problem = problem
        graph = problem.graph
        max_weight = max([graph.edges[edge]["weight"] for edge in graph.edges])

        self.ori_node_feature = []
        self.ori_edge_index = [[], []]
        self.ori_edge_weight = []
        self.ori_edge_index_list = []

        for node in graph.nodes:
            self.ori_node_feature.append([len(graph.adj[node])])
            for neighbor in graph.adj[node]:
                self.ori_edge_index[0].append(node)
                self.ori_edge_index[1].append(neighbor)
                self.ori_edge_weight.append(
                    graph.adj[node][neighbor]["weight"] / max_weight
                )
                self.ori_edge_index_list.append(var2idx[nodes2edge(node, neighbor)])

        self.ori_node_feature = np.asarray(self.ori_node_feature)
        self.ori_edge_index = np.asarray(self.ori_edge_index)
        self.ori_lens = np.array([self.ori_node_feature.shape[0], self.ori_edge_index.shape[1]])

    def get_original_graph_representation(self, solution: np.array, lb: np.array, ub: np.array, *args, **kwargs) -> Dict[str, np.array]:
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

        return dict(
            node_feature=self.ori_node_feature,
            edge_index=self.ori_edge_index,
            edge_feature=ori_edge_feature,
            lens=self.ori_lens,
        )

    def get_solution_representation(self, solution_graph: nx.Graph, *args, **kwargs) -> PseudoTorchData:
        return super().get_solution_representation(solution_graph, *args, **kwargs)


# write a new class for cycle state extractor with modified features
class CycleStateExtractor2(BaseStateExtractor):
    def get_state_representation(self, cb, solution: np.ndarray = None, solution_graph: nx.Graph = None):
        features = [
            # new features related to the root node
            sum(cb.root_node.cut_distances) / cb.root_node.nb_cuts if cb.root_node.nb_cuts > 0 else 0,
            sum(cb.root_node.obj_improvements) / len(cb.root_node.obj_improvements) if cb.root_node.nb_cuts > 0 else 0,
            np.log10(cb.root_node.nb_cut_rounds),
            # get a local IP relative gap for the current node, that is, the relative gap between the objective value
            # of the current node and the best objective value found so far
            min(1, cb.curr_state.obj / (cb.get_incumbent_objective_value() + TOLERANCE)) if cb.has_incumbent() else 1,
            # 12 old features related to the current node
            np.log10(cb.curr_state.depth),
            len(cb.cut_nodes) / cb.processed_nodes,
            np.linalg.norm(cb.curr_state.solution - cb.prev_state.solution),
            abs(cb.curr_state.obj - cb.prev_state.obj),
            max(np.log10(cb.curr_node.nb_cuts), 0),
            sum(cb.curr_node.cut_distances) / cb.curr_node.nb_cuts if cb.curr_node.nb_cuts > 0 else 0,
            sum(cb.curr_node.obj_improvements) / len(cb.curr_node.obj_improvements)
            if len(cb.curr_node.obj_improvements) > 0 else 0,
            max(np.log10(max(cb.curr_node.id - cb.last_cut_node.id, TOLERANCE)), 0),
            sum(cb.last_cut_node.cut_distances) / cb.last_cut_node.nb_cuts if cb.last_cut_node.nb_cuts > 0 else 0,
            sum(cb.last_cut_node.obj_improvements) / len(cb.last_cut_node.obj_improvements)
            if len(cb.last_cut_node.obj_improvements) > 0 else 0,
            sum(cb.last_cut_node.last_cut_distances) / len(cb.last_cut_node.last_cut_distances)
            if len(cb.last_cut_node.last_cut_distances) > 0 else 0,
            cb.last_cut_node.obj_improvements[-1] if len(cb.last_cut_node.obj_improvements) > 0 else 0,
        ]

        return features


class GraphormerCycleStateExtractor(CycleStateExtractor):
    def get_solution_representation(self, solution_graph: nx.Graph, *args, **kwargs) -> PseudoTorchData:
        item = super().get_solution_representation(solution_graph, *args, **kwargs)
        node_feature, edge_index, edge_feature = item.x, item.edge_index, item.edge_attr
        N = node_feature.shape[0]
        edge_index = np.asarray(edge_index)
        edge_feature = np.asarray(edge_feature)

        adj = nx.adjacency_matrix(solution_graph).todense()
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


class EmbedGraphormerCycleStateExtractor(GraphormerCycleStateExtractor):
    def get_statistic_representation(self, cb, *args, **kwargs):
        statistic = {
            "categorical": {"depth": cb.curr_state.depth, "cut_node_dist": cb.curr_node.id - cb.last_cut_node.id},
            "continuous": [
                cb.nb_generating_cuts / cb.curr_state.id,
                np.linalg.norm(
                    cb.curr_state.solution.toarray() - cb.prev_state.solution.toarray()) / cb.max_solution_dist,
                abs(cb.curr_state.obj - cb.prev_state.obj) / cb.max_obj_improvement,
                cb.curr_node.nb_cuts / cb.root_node.nb_cuts if cb.root_node.nb_cuts > 0 else 0,
                sum(cb.curr_node.cut_distances) / cb.curr_node.nb_cuts if cb.curr_node.nb_cuts > 0 else 0,
                sum(cb.curr_node.obj_improvements) / (len(cb.curr_node.obj_improvements) * cb.max_obj_improvement)
                if len(cb.curr_node.obj_improvements) > 0 else 0,
                sum(cb.last_cut_node.cut_distances) / cb.last_cut_node.nb_cuts if cb.last_cut_node.nb_cuts > 0 else 0,
                sum(cb.last_cut_node.obj_improvements) / (
                            len(cb.last_cut_node.obj_improvements) * cb.max_obj_improvement)
                if len(cb.last_cut_node.obj_improvements) > 0 else 0,
                sum(cb.last_cut_node.last_cut_distances) / len(cb.last_cut_node.last_cut_distances)
                if len(cb.last_cut_node.last_cut_distances) > 0 else 0,
                cb.last_cut_node.obj_improvements[-1] / cb.max_obj_improvement if len(
                    cb.last_cut_node.obj_improvements) > 0 else 0,
            ]
        }

        return statistic
