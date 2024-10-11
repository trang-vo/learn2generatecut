"""
The separation routine for the subtour inequalities in the TSP problem
"""

from typing import *

import networkx as nx
import numpy as np

from solvers.callbacks.separators.base import Separator
from constant import TOLERANCE
from utils import nodes2edge


class SubtourSeparator(Separator):
    def __init__(self, var2idx: Dict[Any, int], **kwargs) -> None:
        super().__init__(var2idx)
        self.cut_type = "subtour"

    def create_general_support_graph(self, solution: np.array) -> nx.Graph:
        return self.create_support_graph(solution)

    def create_support_graph(self, solution: np.array) -> nx.Graph:
        support_graph = nx.Graph()

        nz_indices = np.where(solution > TOLERANCE)[0].tolist()
        for idx in nz_indices:
            support_graph.add_edge(*self.idx2var[idx], weight=solution[idx])

        return support_graph

    def get_lazy_constraints(
        self, support_graph: nx.Graph
    ) -> List[Tuple[List[int], List[int], str, int]]:
        constraints: List[Tuple[List[int], List[int], str, int]] = []
        components = list(nx.connected_components(support_graph))

        if len(components) > 1:
            for cc in components:
                edges = support_graph.subgraph(cc).edges
                vars = [self.var2idx[nodes2edge(*edge)] for edge in edges]
                coefs = [1] * len(vars)
                sense = "L"
                constraints.append((vars, coefs, sense, len(edges) - 1))

        return constraints

    def get_user_cuts(
        self, support_graph: nx.Graph
    ) -> List[Tuple[List[int], List[int], str, int]]:
        tree: nx.Graph = nx.gomory_hu_tree(support_graph, capacity="weight")
        cuts: List[Tuple[List[int], List[int], str, int]] = []

        for edge in tree.edges:
            if 2 - tree.edges[edge]["weight"] > TOLERANCE:
                w = tree.edges[edge]["weight"]
                tree.remove_edge(*edge)
                V1 = nx.node_connected_component(tree, edge[0])
                V2 = nx.node_connected_component(tree, edge[1])
                tree.add_edge(edge[0], edge[1], weight=w)

                vars = []
                for u in V1:
                    for v in V2:
                        vars.append(self.var2idx[nodes2edge(u, v)])
                coefs = [1] * len(vars)
                sense = "G"

                cuts.append((vars, coefs, sense, 2))

        return cuts
