from abc import ABC
from typing import *

import numpy as np
import networkx as nx

from problems.base import Problem
from solvers.cplex_api import UserCutCallback
from utils import PseudoTorchData
from constant import TOLERANCE


class BaseStateExtractor(ABC):
    def __init__(self, component_list: List[str], mdp_type: str = "original", padding: bool = False, *args, **kwargs):
        self.component_list = component_list
        self.mdp_type = mdp_type
        self.padding = padding

        self.max_nb_edges = 0
        self.problem = None

    def initialize_original_graph(self, problem: Problem, var2idx: Dict[Tuple[int, int], int], *args, **kwargs) -> None:
        raise NotImplementedError

    def get_solution_representation(self, solution_graph: nx.Graph, *args, **kwargs) -> PseudoTorchData:
        node_feature = np.array([[len(solution_graph.adj[node])] for node in solution_graph.nodes])
        edge_feature = []
        edge_index = [[], []]

        node_label = {}
        for node in solution_graph.nodes:
            node_label[node] = len(node_label)
        solution_graph = nx.relabel_nodes(solution_graph, node_label)

        for node in solution_graph.nodes:
            for neighbor in solution_graph.adj[node]:
                edge_index[0].append(node)
                edge_index[1].append(neighbor)
                edge_feature.append([solution_graph.adj[node][neighbor]["weight"]])

        item = PseudoTorchData()
        item.x = node_feature.astype(np.float64)
        item.edge_index = np.asarray(edge_index, dtype=np.int16)
        item.edge_attr = np.asarray(edge_feature, dtype=np.float64)

        return item

    def get_original_graph_representation(self, solution: np.array, lb: np.array, ub: np.array, *args,
                                          **kwargs) -> Dict[str, np.array]:
        raise NotImplementedError

    def get_statistic_representation(self, cb: UserCutCallback, *args, **kwargs) -> List[Union[float, int]]:
        if self.mdp_type == "simplified":
            statistic = [
                1 / (1 + 1 / cb.curr_state.depth) - 0.5,
                cb.nb_generating_cuts / cb.curr_state.id,
                np.linalg.norm(
                    cb.curr_state.solution.toarray() - cb.prev_state.solution.toarray()) / cb.max_solution_dist,
                abs(cb.curr_state.obj - cb.prev_state.obj) / cb.max_obj_improvement,
                cb.curr_node.nb_cuts / cb.root_node.nb_cuts if cb.root_node.nb_cuts > 0 else 0,
                sum(cb.curr_node.cut_distances) / cb.curr_node.nb_cuts if cb.curr_node.nb_cuts > 0 else 0,
                sum(cb.curr_node.obj_improvements) / (len(cb.curr_node.obj_improvements) * cb.max_obj_improvement)
                if len(cb.curr_node.obj_improvements) > 0 else 0,
                max(np.log10(max(cb.curr_node.id - cb.last_cut_node.id, TOLERANCE)), 0),
                sum(cb.last_cut_node.cut_distances) / cb.last_cut_node.nb_cuts if cb.last_cut_node.nb_cuts > 0 else 0,
                sum(cb.last_cut_node.obj_improvements) / (
                            len(cb.last_cut_node.obj_improvements) * cb.max_obj_improvement)
                if len(cb.last_cut_node.obj_improvements) > 0 else 0,
                sum(cb.last_cut_node.last_cut_distances) / len(cb.last_cut_node.last_cut_distances)
                if len(cb.last_cut_node.last_cut_distances) > 0 else 0,
                cb.last_cut_node.obj_improvements[-1] / cb.max_obj_improvement if len(
                    cb.last_cut_node.obj_improvements) > 0 else 0,
            ]
        elif self.mdp_type == "original":
            processed_leaves: int = cb.get_num_nodes()
            remain_leaves: int = cb.get_num_remaining_nodes()
            total_leaves: int = processed_leaves + remain_leaves

            assert "lb" in kwargs and "ub" in kwargs, "Lower bound and upper bound must be provided"
            lb: np.array = kwargs["lb"]
            ub: np.array = kwargs["ub"]

            statistic = [
                cb.curr_state.depth / len(self.problem.graph.nodes),
                cb.has_incumbent(),
                cb.curr_state.gap,
                sum(np.where(abs(cb.curr_state.solution.toarray()[0] - 1) < TOLERANCE, 1, 0)) / len(
                    self.problem.graph.nodes),
                cb.curr_state.obj / cb.get_cutoff(),
                remain_leaves / total_leaves,
                processed_leaves / total_leaves,
                np.sum(lb) / len(self.problem.graph.nodes),
                (lb.shape[0] - np.sum(lb)) / cb.curr_state.solution.toarray().shape[0],
                np.sum(ub) / cb.curr_state.solution.toarray().shape[0],
                (ub.shape[0] - np.sum(ub)) / cb.curr_state.solution.toarray().shape[0],
                sum(np.where(abs(lb - ub) < TOLERANCE, 1, 0)) / cb.curr_state.solution.toarray().shape[0],
            ]
        else:
            ValueError("Invalid MDP type")

        return statistic

    def get_state_representation(
            self,
            callback,
            solution: np.ndarray = None,
            solution_graph: nx.Graph = None,
    ) -> Dict[str, Any]:
        state_rep = {}
        lb: np.array = np.asarray(callback.get_lower_bounds())
        ub: np.array = np.asarray(callback.get_upper_bounds())

        if "solution" in self.component_list:
            state_rep["solution"] = self.get_solution_representation(solution_graph=solution_graph)
        if "statistic" in self.component_list:
            state_rep["statistic"] = self.get_statistic_representation(callback, lb=lb, ub=ub)
        if "problem" in self.component_list:
            state_rep["problem"] = self.get_original_graph_representation(solution=solution, lb=lb, ub=ub)

        return state_rep

    def get_cut_detector_solution_representation(self, solution_graph: nx.Graph) -> Dict[str, np.ndarray]:
        raise NotImplementedError
