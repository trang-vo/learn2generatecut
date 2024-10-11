import _pickle
import os.path
import random
import re
import shutil
import subprocess
from glob import glob
from typing import *

import networkx as nx
import typer
import numpy as np

from solvers.cplex_api import cplex

app = typer.Typer()


@app.command()
def divide_tsp_instances(min_node: int, max_node: int, ngroups: int):
    output = {i: [] for i in range(ngroups)}
    paths = glob("../data/tsplib/*.tsp")
    size_instances = {}
    for path in paths:
        nNode = get_num_nodes_from_path(path)
        if nNode < min_node or nNode >= max_node:
            continue

        size_instances[path] = nNode

    size_instances = {k: v for k, v in sorted(size_instances.items(), key=lambda item: item[1])}
    m = 0
    for path in size_instances:
        output[int(m % ngroups)].append(path)
        m += 1

    with open("tsp_group_{}_{}_{}.p".format(min_node, max_node, ngroups), "wb") as file:
        _pickle.dump(output, file)


@app.command()
def get_num_nodes_from_path(path: str):
    tmp = path.split("/")
    nums = re.findall(r'\d+', tmp[-1])
    return int(nums[0])


def nodes2edge(u, v):
    return min(u, v), max(u, v)


def distance_solution_cuts(solution: np.array, vars: List[int], coefs: List[float], rhs: float):
    return abs(sum([coefs[i] * solution[var_idx] for i, var_idx in enumerate(vars)]) - rhs) / np.sqrt(len(coefs))


def get_objective_parallelism(obj_coefs: np.array, vars: List[int], coefs: List[float], *args):
    return abs(np.sum(obj_coefs[vars])) / (np.sqrt(len(coefs)) * np.linalg.norm(obj_coefs))


class ResultWriter:
    def __init__(self, path: str, solver: cplex.Cplex, callback, kws: List[str], external_kws: List[str] = None,
                 separator: str = ","):
        self.path = path
        self.solver = solver
        self.callback = callback
        self.kws = kws
        self.external_kws = external_kws if external_kws is not None else []
        self.separator = separator

        self.available_kws = {
            "instanceName": "The name of the instance",
            "gap": "The best IP relative gap",
            "nNodes": "The number of nodes in the search tree",
            "nCuts": "The number of added cuts from fractional solutions",
            "nSepa": "The number of solved separation problems",
            "action_1": "The number of times deciding to generate cuts",
            "action_0": "The number of times deciding to branch",
            "nSepaWithCuts": "The number of separation problems that can provide violated constraints",
            "rootEfficacy": "The average efficacy of cuts generated at the root node",
            "rootObjParallelism": "The average objective parallelism of cuts generated at the root node",
        }

        for kw in self.kws:
            if kw not in self.available_kws:
                raise ValueError("{} is not available".format(kw))

        if not os.path.isfile(self.path):
            title = self.separator.join(self.kws + self.external_kws) + "\n"
            with open(self.path, "w") as file:
                file.write(title)

    def write(self, external_kws: Dict[str, Any] = None):
        msg = []
        for kw in self.kws:
            value = self._get_kw_value(kw)
            if isinstance(value, float):
                msg.append("{:.4f}".format(value))
            else:
                msg.append(str(value))

        for kw in self.external_kws:
            if kw not in external_kws:
                raise KeyError("Provide a value of the keyword {}".format(kw))
            value = external_kws[kw]
            if isinstance(value, float):
                msg.append("{:.4f}".format(value))
            else:
                msg.append(str(value))

        msg = self.separator.join(msg) + "\n"
        with open(self.path, "a") as file:
            file.write(msg)

    def get_available_keywords(self):
        return list(self.available_kws.keys())

    def get_keyword_description(self, kw: str):
        return self.available_kws[kw]

    def _get_kw_value(self, kw: str):
        try:
            if kw in ["instanceName"]:
                return self.solver.problem.graph.name
            elif kw in ["gap"]:
                return self.solver.solution.MIP.get_mip_relative_gap() * 100
            elif kw in ["nNodes"]:
                return self.callback.processed_nodes
            elif kw in ["nCuts"]:
                return self.callback.total_cuts
            elif kw in ["nSepa", "action_1"]:
                return self.callback.actions[1]
            elif kw in ["action_0"]:
                return self.callback.actions[0]
            elif kw in ["nSepaWithCuts"]:
                return self.callback.portion_cuts["with_cut"]
            elif kw in ["rootEfficacy"]:
                return self.callback.root_efficacy
            elif kw in ["rootObjParallelism"]:
                return self.callback.root_obj_parallelism
        except Exception as e:
            self.solver.logger.write("Error in ResultWriter: {}\n".format(e))
            return None


def copy_file_to_folder(source_file, destination_folder, new_file_name=None):
    try:
        if new_file_name:
            destination_file = os.path.join(destination_folder, new_file_name)
        else:
            destination_file = os.path.join(destination_folder, os.path.basename(source_file))

        shutil.copy(source_file, destination_file)
        print(f"File '{source_file}' copied to '{destination_file}'.")
    except FileNotFoundError:
        print("Source file not found.")
    except PermissionError:
        print(
            "Permission denied. Make sure you have read access to the source file and write access to the destination folder.")


def modify_maxcut_instance(graph: nx.Graph, change_percent: float, criterion: List[str] = ("weight")):
    assert 0 <= change_percent <= 1, "The percentage must be in the range of 0 to 1"
    nb_modified_edges = int(change_percent * len(graph.edges))
    modified_edges = random.sample(list(graph.edges), nb_modified_edges)

    for edge in modified_edges:
        weight = graph.edges[edge]["weight"]
        if "weight" in criterion:
            graph.edges[edge]["weight"] = -1 * weight

        if "edge" in criterion:
            graph.remove_edge(*edge)
            nodes = list(graph.nodes)
            new_edge = (random.choice(nodes), random.choice(nodes))
            while graph.has_edge(*new_edge):
                new_edge = (random.choice(nodes), random.choice(nodes))
            graph.add_edge(*new_edge, weight=weight)

    return graph


def write_graph_to_maxcut_file(graph, path):
    lines = [f'{len(graph.nodes)} {len(graph.edges)}']
    for edge in graph.edges:
        lines.append(f'{edge[0]} {edge[1]} {graph.edges[edge]["weight"]}')
    with open(path, "w") as file:
        file.write("\n".join(lines))


class PseudoTorchData:
    def __init__(self):
        self.x = None
        self.edge_index = None
        self.edge_attr = None
        self.batch = None
        self.in_degree = None
        self.out_degree = None
        self.attn_bias = None
        self.attn_edge_type = None
        self.spatial_pos = None
        self.edge_input = None


def get_branch_info():
    try:
        # Get the current branch name
        branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()

        # Get the commit SHA of the latest commit on the current branch
        commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

        return branch_name, commit_sha

    except Exception as e:
        return None, None


class StateRecorder:
    def __init__(self, *args, **kwargs):
        self.nb_cuts = None
        self.gap = None
        self.time = None
        self.obj = None
        self.cutoff = None
        self.depth = 0
        self.solution = None
        self.node_id = 0
        self.id = None
        self.cut_distances = []
        for kw, value in kwargs.items():
            setattr(self, kw, value)


class NodeRecorder:
    def __init__(self, **kwargs):
        self.nb_separation: int = 0
        self.nb_cuts: int = 0
        self.id: int = -1
        self.depth: int = 0
        # Compute the average cut's quality measures
        self.cut_distances = []
        # Compute the cut's quality measures at the last round of cut generation at the node
        self.last_cut_distances = []
        self.obj_improvements = []
        self.nb_cut_rounds = 0
        self.nb_visited = 0


if __name__ == "__main__":
    app()


