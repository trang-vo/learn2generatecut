from .base import Problem

import tsplib95
import networkx as nx


class TSPProblem(Problem):
    def __init__(self, path: str, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)
        self.type = "tsp"

    def read_file(self, path: str, *args, **kwargs):
        """Read a TSP instance from a file in the tsplib format. Save the instance in a nx.graph object."""
        problem = tsplib95.load(path)
        self.graph: nx.Graph = problem.get_graph(normalize=True)

        removed_edges = []
        for edge in self.graph.edges:
            if edge[0] == edge[1]:
                removed_edges.append(edge)

        for edge in removed_edges:
            self.graph.remove_edge(*edge)

        self.graph = self.graph.to_undirected()
