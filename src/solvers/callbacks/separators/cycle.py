"""
The separation routine for the cycle inequalities in the Max-Cut problem
"""

from typing import *

import networkx as nx
import numpy as np

from .base import Separator
from constant import TOLERANCE
from utils import nodes2edge


class CycleSeparator(Separator):
    def __init__(self, var2idx: Dict[Any, int], **kwargs) -> None:
        super().__init__(var2idx)
        self.cut_type = "cycle"
        assert "origin_graph" in kwargs, "Require the origin graph to initialize Cycle separator"
        self.origin_graph = kwargs["origin_graph"]

    def create_support_graph(self, solution: np.array):
        support_graph = nx.Graph()

        for idx, (u, v) in self.idx2var.items():
            if abs(solution[idx]) < TOLERANCE:
                support_graph.add_edges_from(
                    [(u, v), (str(u), str(v))], weight=max(0, solution[idx]), label=0
                )
                continue
            if abs(solution[idx] - 1) < TOLERANCE:
                support_graph.add_edges_from(
                    [(u, str(v)), (str(u), v)],
                    weight=max(0, 1 - solution[idx]),
                    label=1,
                )
                continue
            support_graph.add_edges_from(
                [(u, v), (str(u), str(v))], weight=max(0, solution[idx]), label=0
            )
            support_graph.add_edges_from(
                [(u, str(v)), (str(u), v)], weight=max(0, 1 - solution[idx]), label=1
            )

        return support_graph

    def _get_violated_constraints(self, support_graph: nx.Graph):
        cuts = []

        for node in self.origin_graph.nodes:
            try:
                path = nx.shortest_path(support_graph, node, str(node), weight="weight")
                node_path = set([int(n) for n in path])
                edge_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                length = sum(
                    [support_graph.edges[edge]["weight"] for edge in edge_path]
                )
                if length + TOLERANCE < 1 and len(node_path) == len(edge_path):
                    expr, coefs = [], []
                    rhs = 0
                    sense = "L"
                    for edge in edge_path:
                        expr.append(
                            self.var2idx[nodes2edge(int(edge[0]), int(edge[1]))]
                        )
                        if support_graph.edges[edge]["label"] == 0:
                            coefs.append(-1)
                        else:
                            coefs.append(1)
                            rhs += 1
                    cuts.append((expr, coefs, sense, rhs - 1))
            except nx.exception.NetworkXNoPath:
                continue

        return cuts

    def get_lazy_constraints(self, support_graph: nx.Graph):
        return self._get_violated_constraints(support_graph)

    def get_user_cuts(self, support_graph: nx.Graph):
        return self._get_violated_constraints(support_graph)
