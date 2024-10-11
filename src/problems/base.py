from typing import *

import networkx as nx


class Problem:
    def __init__(self, path: str, *args, **kwargs) -> None:
        self.type = None
        self.graph = nx.Graph()
        self.read_file(path)

    def read_file(self, path: str, *args, **kwargs):
        """Read a problem from a file"""
        raise NotImplementedError
