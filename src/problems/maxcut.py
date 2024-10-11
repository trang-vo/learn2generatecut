import networkx as nx

from .base import Problem
from utils import nodes2edge


class MaxcutProblem(Problem):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.type = "maxcut"

    def read_file(self, path: str):
        """Read a Maxcut instance from a file in the rudy format. Save the instance in a nx.graph object."""
        with open(path, "r") as file:
            data = file.readlines()

        self.graph = nx.Graph()
        self.graph.graph["name"] = path.split("/")[-1]
        for line in data[1:]:
            tmp = line.strip("\n").split(" ")
            edge = nodes2edge(int(tmp[0]), int(tmp[1]))
            self.graph.add_edge(*edge, weight=int(tmp[2]))

        label_node = {}
        for node in self.graph.nodes:
            label_node[node] = len(label_node)
        self.graph = nx.relabel_nodes(self.graph, label_node)
