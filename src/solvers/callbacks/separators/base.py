from typing import *

import numpy as np
import networkx as nx

from constant import TOLERANCE


class Separator:
    def __init__(self, var2idx: Dict[Any, int]) -> None:
        self.cut_type = ""
        self.var2idx = var2idx
        self.idx2var = {idx: var for var, idx in var2idx.items()}

    def create_general_support_graph(self, solution: np.array) -> nx.Graph:
        """
        Create a general-purpose support graph to represent a solution.
        In particular, given a solution, we create a graph as follows:
            - For each variable x_i, if x_i > 0, we add an edge (u, v) corresponding to x_i with weight x_i
        """
        graph = nx.Graph()

        nz_indices = np.where(solution > TOLERANCE)[0].tolist()
        for idx in nz_indices:
            graph.add_edge(*self.idx2var[idx], weight=solution[idx])

        return graph

    def create_support_graph(self, solution: np.array) -> nx.Graph:
        """Create a support graph to find violated constraints"""
        raise NotImplementedError

    def get_lazy_constraints(self, support_graph: nx.Graph):
        """Find constraints of self.cut_type violated by a feasible integer solution (mandatory)"""
        raise NotImplementedError

    def get_user_cuts(self, support_graph: nx.Graph):
        """Find constraints of self.cut_type violated by a fractional solution (optional)"""
        raise NotImplementedError
