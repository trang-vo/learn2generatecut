from typing import *


class Config:
    def __init__(self, config: Dict[Any, Any]) -> None:
        pass


class EnvConfig(Config):
    def __init__(self, config: Dict[Any, Any]) -> None:
        self.instance_size = config["instance_size"]
        self.sup_nNodes = config["sup_nNodes"]
        self.sup_nEdges = config["sup_nEdges"]
        self.sup_node_dim = config["sup_node_dim"]
        self.sup_edge_dim = config["sup_edge_dim"]
        self.ori_nEdges = config["ori_nEdges"]
        self.ori_node_dim = config["ori_node_dim"]
        self.ori_edge_dim = config["ori_edge_dim"]
        self.statistic_dim = config["statistic_dim"]
