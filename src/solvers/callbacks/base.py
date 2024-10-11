import os.path
import sys
from collections import defaultdict
from copy import deepcopy
from time import time
from typing import *

import networkx as nx
import numpy as np
import torch
import tianshou as ts
from scipy.sparse import coo_matrix

from constant import TOLERANCE
from utils import distance_solution_cuts, get_objective_parallelism
from .separators.base import Separator
from .state_extractor.base import BaseStateExtractor
from .common import StateRecorder, NodeRecorder
from ..cplex_api import cplex, LazyConstraintCallback, UserCutCallback


class BaseLazyCallback(LazyConstraintCallback):
    def __call__(self):
        solution = np.asarray(self.get_values())

        support_graph = self.separator.create_support_graph(solution)
        constraints = self.separator.get_lazy_constraints(support_graph)
        for vars, coefs, sense, rhs in constraints:
            self.add(constraint=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        self.separator = separator


class BaseUserCallback(UserCutCallback):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        if self.processed_nodes % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        if self.processed_nodes == 0 and self.root_cuts is not None:
            cuts = next(self.root_cuts, [])

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            return

        s = time()
        self.actions[1] += 1
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
        self.total_cuts += len(cuts)
        if len(cuts) > 0:
            self.portion_cuts["with_cut"] += 1
        else:
            self.portion_cuts["without_cut"] += 1
        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
            self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
        )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

    def set_attribute(self, separator: Separator, *args, **kwargs):
        self.separator: Separator = separator
        self.total_cuts = 0
        self.frequent = kwargs["frequent"] if "frequent" in kwargs else 1
        self.terminal_gap = kwargs["terminal_gap"] if "terminal_gap" in kwargs else 0
        self.logger = kwargs["logger"] if "logger" in kwargs else sys.stdout
        self.processed_nodes = 0
        self.actions = {0: 0, 1: 0}
        self.portion_cuts = {"with_cut": 0, "without_cut": 0}
        self.root_cuts = kwargs["root_cuts"] if "root_cuts" in kwargs else None
        if self.root_cuts is not None:
            print("Loaded root cuts")


class PreprocessUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        if self.skip_root and self.get_num_nodes() == 0:
            return

        curr_depth = self.get_current_node_depth()
        if self.prev_depth > curr_depth:
        # if self.processed_nodes > 0:
            self.abort()
            return

        self.prev_depth = curr_depth

        if self.get_num_nodes() % self.frequent != 0:
            return

        s = time()
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
        self.total_cuts += len(cuts)

        if self.processed_nodes == 0:
            self.root_cuts.append(cuts)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super(PreprocessUserCallback, self).set_attribute(separator, *args, **kwargs)
        self.skip_root = kwargs["skip_root"] if "skip_root" in kwargs else False
        self.prev_depth = 0
        self.root_cuts = []


class RLUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        s = time()
        processed_leaves = self.get_num_nodes()

        # Generate cuts at the root node
        if processed_leaves == 0:
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            ncuts = len(cuts)
            self.start = time()
            self.prev_cuts = len(cuts)
            self.prev_gap = min(self.prev_gap, self.get_MIP_relative_gap())
            self.prev_time = time()
            self.total_time += s - time()
            self.total_cuts += len(cuts)
            msg = "Node 0, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                ncuts, time() - s, self.total_cuts)
            if self.logger is not None:
                self.logger.write(msg)
            else:
                print(msg, end="")
            return

        # Skip generate cuts when the MIP gap is less than 1%
        gap = min(self.prev_gap, self.get_MIP_relative_gap())
        if gap < self.config["terminal_gap"]:
            return

        # Extract state information
        s = time()
        state, support_graph = self.state_extractor.get_state_representation(self)

        for key in state:
            if isinstance(state[key], np.ndarray):
                state[key] = torch.Tensor(np.asarray([state[key]]))

        with torch.no_grad():
            state_feature = self.features_extractor(state)
            action = self.agent(state_feature)[0].argmax().tolist()
            self.actions[action] += 1

            ncuts = -1
            if action == 1:
                s = time()
                cuts = self.separator.get_user_cuts(support_graph)
                for vars, coefs, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

                ncuts = len(cuts)
                self.total_cuts += ncuts
                print("Time to solver separation problem", time() - s)
            self.prev_cuts = ncuts
            self.prev_gap = gap
        t = time() - s

        msg = "Node {}, add {} user cuts, gap {:.2f}, total cuts {}, {}, predict action in {:.4f}s\n".format(
            processed_leaves,
            self.prev_cuts,
            gap * 100 if gap < 1 else -1,
            self.total_cuts,
            self.actions,
            t,
        )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: BaseStateExtractor = kwargs["state_extractor"]
        self.state_extractor.padding = False
        self.features_extractor = kwargs["features_extractor"]
        self.features_extractor.eval()
        self.agent = kwargs["agent"]
        self.agent.eval()
        self.config: Dict[str, Any] = kwargs["config"]

        self.prev_cuts = 0
        self.prev_gap = 1
        self.total_cuts = 0
        self.total_time = 0
        self.actions = {0: 0, 1: 0}
        self.last_state = None


class MLUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        s = time()
        self.processed_nodes = self.get_num_nodes()

        # Generate cuts at the root node
        if self.processed_nodes == 0:
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            ncuts = len(cuts)
            self.start = time()
            self.prev_cuts = len(cuts)
            self.prev_gap = min(self.prev_gap, self.get_MIP_relative_gap())
            self.prev_time = time()
            self.total_time += s - time()
            self.total_cuts += len(cuts)
            msg = "Node 0, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                ncuts, time() - s, self.total_cuts)
            self.logger.write(msg)
            return

        # Skip generate cuts when the MIP gap is less than 1%
        gap = min(self.prev_gap, self.get_MIP_relative_gap())
        if gap < self.config["terminal_gap"]:
            return

        if self.processed_nodes % self.frequent != 0 or (self.only_root and self.processed_nodes > 0):
            return

        print("Use cut detector", self.use_cut_detector, "Use RL agent", self.use_rl_agent)
        if self.use_cut_detector and self.use_rl_agent:
            s = time()
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution=solution)

            support_rep = self.state_extractor.get_cut_detector_solution_representation(support_graph)
            support_batch = np.zeros(support_rep["sup_lens"][0])
            exist_cuts = self.cut_detector.predict(support_rep["sup_node_feature"], support_rep["sup_edge_index"],
                                                   support_rep["sup_edge_feature"], support_batch)
            if not exist_cuts:
                self.logger.write("Node {}, predicts no cuts in {:.4f}s\n".format(self.processed_nodes, time() - s))
                return

            state, support_graph = self.state_extractor.get_state_representation(self, solution=solution,
                                                                                 solution_graph=support_graph)
            
            for key in state:
                if isinstance(state[key], np.ndarray):
                    state[key] = torch.Tensor(np.asarray([state[key]]))

            with torch.no_grad():
                state_feature = self.features_extractor(state)
                action = self.agent(state_feature)[0].argmax().tolist()
                self.actions[action] += 1
                t_action = time() - s

            ncuts = -1
            if action == 1:
                s = time()
                cuts = self.separator.get_user_cuts(support_graph)
                for vars, coefs, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

                ncuts = len(cuts)
                self.total_cuts += ncuts
            self.prev_cuts = ncuts
            self.prev_gap = gap
            t = time() - s

            msg = "Node {}, add {} user cuts in {:.4f}s, gap {:.2f}, total cuts {}, {}, predict action in {:.4f}s\n".format(
                self.processed_nodes,
                self.prev_cuts,
                t,
                gap * 100 if gap < 1 else -1,
                self.total_cuts,
                self.actions,
                t_action,
            )
            self.logger.write(msg)

        elif self.use_cut_detector and not self.use_rl_agent:
            s = time()
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution=solution)

            support_rep = self.state_extractor.get_cut_detector_solution_representation(support_graph)
            support_batch = np.zeros(support_rep["sup_lens"][0])
            exist_cuts = self.cut_detector.predict(support_rep["sup_node_feature"], support_rep["sup_edge_index"],
                                                   support_rep["sup_edge_feature"], support_batch)
            self.logger.write("Predict cut existence in {:.4f}s\n".format(time() - s))
            if not exist_cuts:
                self.logger.write("Node {}, cut detector predicts no cuts\n".format(self.processed_nodes))
                self.actions[0] += 1
            else:
                cuts = self.separator.get_user_cuts(support_graph)
                for vars_, coefs, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars_, coefs), sense=sense, rhs=rhs)
                self.total_cuts += len(cuts)
                self.prev_cuts = len(cuts)
                self.prev_gap = gap
                self.actions[1] += 1
                t = time() - s
                self.logger.write("Node {}, add {} cuts in {:.4f}s\n".format(self.processed_nodes, self.prev_cuts, t))
            return

        elif not self.use_cut_detector and self.use_rl_agent:
            # Extract state information
            s = time()
            state, support_graph = self.state_extractor.get_state_representation(self)

            action = 0
            if not nx.is_connected(support_graph):
                action = 1
            else:
                for key in state:
                    if isinstance(state[key], np.ndarray):
                        state[key] = torch.Tensor(np.asarray([state[key]]))

                with torch.no_grad():
                    state_feature = self.features_extractor(state)
                    action = self.agent(state_feature)[0].argmax().tolist()
                    self.actions[action] += 1

            t_action = time() - s
            ncuts = -1
            if action == 1:
                s = time()
                cuts = self.separator.get_user_cuts(support_graph)
                for vars, coefs, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

                ncuts = len(cuts)
                self.total_cuts += ncuts
            self.prev_cuts = ncuts
            self.prev_gap = gap
            t = time() - s

            msg = "Node {}, add {} user cuts in {:.4f}s, gap {:.2f}, total cuts {}, {}, predict action in {:.4f}s\n".format(
                self.processed_nodes,
                self.prev_cuts,
                t,
                gap * 100 if gap < 1 else -1,
                self.total_cuts,
                self.actions,
                t_action,
            )
            self.logger.write(msg)

        elif not self.use_cut_detector and not self.use_rl_agent:
            self.actions[1] += 1
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
            self.total_cuts += len(cuts)
            msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
            )
            self.logger.write(msg)

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: BaseStateExtractor = kwargs["state_extractor"]
        self.state_extractor.padding = False
        self.config: Dict[str, Any] = kwargs["config"]
        self.only_root: bool = kwargs["only_root"] if "only_root" in kwargs else False

        self.use_cut_detector = False
        if "cut_detector" in kwargs:
            self.use_cut_detector = True
            self.cut_detector = kwargs["cut_detector"]
            # self.cut_detector.eval()


        self.use_rl_agent = True
        if "agent" in kwargs:
            self.use_rl_agent = True
            self.features_extractor = kwargs["features_extractor"]
            self.features_extractor.eval()
            self.agent = kwargs["agent"]
            self.agent.eval()

        self.prev_cuts = 0
        self.prev_gap = 1
        self.total_cuts = 0
        self.total_time = 0
        self.actions = {0: 0, 1: 0}
        self.last_state = None


class RandomUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()

        if self.processed_nodes % self.frequent != 0:
            return

        if self.processed_nodes == 0:
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            return

        action = np.random.randint(0, 2)
        # action = 1
        if action == 0:
            return

        # if self.actions[1] >= self.sepa_limit:
        #     return

        self.actions[action] += 1
        s = time()
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

        self.total_cuts += len(cuts)
        if len(cuts) > 0:
            self.sepa_have_cuts += 1
        self.sepa_time += time() - s

        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
            self.processed_nodes, len(cuts), time() - s, self.total_cuts
        )
        self.logger.write(msg)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.skip_root = kwargs["skip_root"] if "skip_root" in kwargs else False
        self.sepa_limit = kwargs["sepa_limit"] if "sepa_limit" in kwargs else 200
        self.sepa_time = 0
        self.sepa_have_cuts = 0


class MiningUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        if self.get_num_nodes() % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        if self.obj_coefs is None:
            self.obj_coefs = np.asarray(self.get_objective_coefficients())

        s = time()
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)
        self.actions[1] += 1

        if len(cuts) > 0:
            self.portion_cuts["with_cut"] += 1
        else:
            self.portion_cuts["without_cut"] += 1

        if self.processed_nodes == 0:
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
                self.efficacy.append(distance_solution_cuts(solution, vars, coefs, rhs))
                self.obj_parallelism.append(get_objective_parallelism(self.obj_coefs, vars, coefs))
        else:
            if not self.report:
                self.report = True
                self.logger.write("The average efficacy is {:.4f}\n".format(sum(self.efficacy) / len(self.efficacy)))
                self.logger.write("The average objective parallelism is {:.4f}\n".format(
                    sum(self.obj_parallelism) / len(self.obj_parallelism)))
                self.root_efficacy = sum(self.efficacy) / len(self.efficacy)
                self.root_obj_parallelism = sum(self.obj_parallelism) / len(self.obj_parallelism)
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
            self.processed_nodes, len(cuts), time() - s, self.total_cuts
        )
        self.logger.write(msg)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.zero_cut = 0
        self.rewards = []
        self.state_time = []
        self.max_bonus = 1
        self.explored_nodes = defaultdict(int)
        self.prev_obj = 0
        self.efficacy = []
        self.obj_parallelism = []
        self.directed_cutoff = []
        self.obj_coefs = None
        self.report = False
        self.root_efficacy = -1
        self.root_obj_parallelism = -1


class RecordRootCutCallback(BaseUserCallback):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        if self.processed_nodes % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        self.actions[1] += 1
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

        if self.processed_nodes == 0:
            self.root_cuts.append(cuts)

        if self.processed_nodes > 0:
            self.abort()

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super(RecordRootCutCallback, self).set_attribute(separator, *args, **kwargs)
        self.root_cuts = []


class TestUserCallback(UserCutCallback):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        if self.processed_nodes % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        s = time()
        self.actions[1] += 1

        if self.processed_nodes == 0:
            cuts = next(self.root_cuts, [])
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
            print("At node 0, add {} user cuts".format(len(cuts)))
            return

        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
            self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
        )
        self.logger.write(msg)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        self.separator: Separator = separator
        self.total_cuts = 0
        self.frequent = kwargs["frequent"] if "frequent" in kwargs else 1
        self.terminal_gap = kwargs["terminal_gap"] if "terminal_gap" in kwargs else 0
        self.logger = kwargs["logger"] if "logger" in kwargs else sys.stdout
        self.processed_nodes = 0
        self.actions = {0: 0, 1: 0}
        self.sepa_have_cuts = 0
        self.sepa_time = 0
        self.root_cuts = kwargs["root_cuts"]
        self.add_root = False


class SkipFactorUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        if self.processed_nodes == 0:
            s = time()
            solution = np.asarray(self.get_values())
            if self.root_cuts is not None:
                cuts = next(self.root_cuts, [])
                print(f"Loaded {len(cuts)} root cuts")
            else:
                support_graph = self.separator.create_support_graph(solution)
                cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
                self.distance_cutoff.append(distance_solution_cuts(solution, vars, coefs, rhs))

            self.total_root_cuts += len(cuts)
            self.total_cuts += len(cuts)
            msg = "Node 0, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                len(cuts), time() - s, self.total_cuts)
            self.logger.write(msg)
            return
        else:
            '''
                Set skip factor k = min(KMAX, ceil(f / c.d.log_{10}p)) 
                where: f is the number of cuts generated at the root node, d is the average distance cutoff, 
                KMAX = 32, c = 5, p is the number of variables
            '''
            if self.frequent is None:
                c = 1000
                d = sum(self.distance_cutoff) / len(self.distance_cutoff)
                print(self.total_root_cuts, c, d, np.log10(len(self.separator.var2idx)))
                print(self.total_root_cuts / (c * d * np.log10(len(self.separator.var2idx))))
                self.frequent = min(32, int(self.total_root_cuts / (c * d * np.log10(len(self.separator.var2idx)))))
                self.logger.write("Automatic skip factor is {}\n".format(self.frequent))

        if self.get_num_nodes() % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        s = time()
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
        self.total_cuts += len(cuts)
        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
            self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
        )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.total_root_cuts = 0
        self.skip_root = kwargs["skip_root"] if "skip_root" in kwargs else False
        self.distance_cutoff = []
        self.frequent = None


class RainbowUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        self.processed_nodes = self.get_num_nodes()
        node_data = deepcopy(self.get_node_data())

        if node_data is None:
            if self.processed_nodes == 0:
                node_data = {"curr_state": StateRecorder(), "parent_node": NodeRecorder(), "curr_node": NodeRecorder(),
                             "last_cut_node": NodeRecorder(), "nb_generating_cuts": 0}
            elif self.processed_nodes != 0:
                raise Exception(f"Node data is None at the non-root node {self.processed_nodes}")

        # Create the state and node recorders
        self.prev_state, self.last_cut_node = node_data["curr_state"], node_data["last_cut_node"]
        self.nb_generating_cuts = node_data["nb_generating_cuts"]
        self.curr_state, self.curr_node = self.initialize_state_and_node_recorders(node_data)

        if self.prev_state.node_id == self.curr_state.node_id and self.prev_state.obj is not None:
            self.curr_node.obj_improvements.append(abs(self.prev_state.obj - self.curr_state.obj))

        self.curr_node.nb_visited += 1

        solution = np.asarray(self.get_values())
        self.curr_state.solution = coo_matrix(solution)

        if self.processed_nodes == 0:
            self.solve_root_node(solution)
            return

        solution_graph = self.separator.create_general_support_graph(solution)
        state_representation = self.state_extractor.get_state_representation(self, solution, solution_graph)

        msg = [
            f"Node {self.prev_state.node_id} (t = {self.prev_state.id})",
            f"add {self.prev_state.nb_cuts} user cuts, gap {self.prev_state.gap:.2f}",
            f"total cuts {self.total_cuts}, actions {self.actions}"
        ]
        self.logger.write(", ".join(msg) + "\n")

        exist_cuts = True
        if self.use_cut_detector:
            s = time()
            support_graph = self.separator.create_support_graph(solution=solution)
            support_rep = self.state_extractor.get_cut_detector_solution_representation(support_graph)
            support_batch = np.zeros(support_rep["sup_lens"][0])
            exist_cuts = self.cut_detector.predict(support_rep["sup_node_feature"], support_rep["sup_edge_index"],
                                                   support_rep["sup_edge_feature"], support_batch)
            self.logger.write("Predict cut existence in {:.4f}s\n".format(time() - s))
            if not exist_cuts:
                self.logger.write("Node {}, cut detector predicts no cuts\n".format(self.processed_nodes))
                self.actions[0] += 1

        action = 0
        if exist_cuts:
            data = ts.data.batch.Batch({"obs": ts.data.batch.Batch([state_representation]), "info": {}})
            action = self.policy(data)["act"][0]

        if self.curr_node.nb_cuts > 0:
            self.last_cut_node = self.curr_node
            self.cut_nodes.add(self.curr_node.id)

        # Perform the action and update the node data
        self.curr_state.time = time()
        self.curr_state.nb_cuts = -1
        self.actions[action] += 1
        if action == 1:
            self.curr_node.last_cut_distances = []
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
                cut_dist = distance_solution_cuts(solution, vars, coefs, rhs)
                self.curr_node.cut_distances.append(cut_dist)
                self.curr_node.last_cut_distances.append(cut_dist)

            self.curr_state.nb_cuts = len(cuts)
            self.curr_node.nb_cuts += len(cuts)
            self.total_cuts += self.curr_state.nb_cuts
            self.nb_generating_cuts += 1

        self.prev_depth = self.curr_state.depth
        self.set_node_data({"curr_state": self.curr_state, "parent_node": node_data["parent_node"],
                            "last_cut_node": self.last_cut_node, "curr_node": self.curr_node,
                            "nb_generating_cuts": self.nb_generating_cuts})

    def initialize_state_and_node_recorders(self, node_data: Dict[str, Any]) -> Tuple[StateRecorder, NodeRecorder]:
        curr_state = StateRecorder()
        curr_state.depth = self.get_current_node_depth()
        curr_state.obj = self.get_objective_value()
        curr_state.gap = min(self.get_MIP_relative_gap(), 999)
        curr_state.id = node_data["curr_state"].id + 1 if node_data["curr_state"].id is not None else 0
        curr_state.node_id = self.processed_nodes

        curr_node = node_data["curr_node"]
        if self.processed_nodes != curr_node.id:
            curr_node = NodeRecorder()
            curr_node.id = self.processed_nodes
            curr_node.depth = self.get_current_node_depth()

        return curr_state, curr_node

    def solve_root_node(self, solution: np.ndarray):
        if self.root_cuts is not None:
            cuts = next(self.root_cuts, [])
        else:
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
            cut_dist = distance_solution_cuts(solution, vars, coefs, rhs)
            self.curr_node.cut_distances.append(cut_dist)
            self.curr_state.cut_distances.append(cut_dist)

        self.curr_node.nb_cut_rounds += 1
        self.curr_node.nb_cuts += len(cuts)
        if self.prev_state.node_id == self.curr_state.node_id and self.prev_state.obj is not None:
            obj_improvement = abs(self.prev_state.obj - self.curr_state.obj)
            self.curr_node.obj_improvements.append(obj_improvement)
            self.max_obj_improvement = max(self.max_obj_improvement, obj_improvement)

            solution_dist = np.linalg.norm(self.curr_state.solution.toarray() - self.prev_state.solution.toarray())
            self.max_solution_dist = max(self.max_solution_dist, solution_dist)

            self.max_nb_cuts = max(self.max_nb_cuts, len(cuts))

        self.curr_state.nb_cuts = len(cuts)
        self.curr_state.id = 0
        if len(cuts) > 0:
            self.cut_nodes.add(self.curr_node.id)
        self.set_node_data({"curr_state": self.curr_state, "parent_node": NodeRecorder(), "curr_node": self.curr_node,
                            "last_cut_node": self.curr_node, "nb_generating_cuts": self.nb_generating_cuts})

        self.root_node = self.curr_node

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: BaseStateExtractor = kwargs["state_extractor"]
        self.config: Dict[str, Any] = kwargs["config"]
        self.policy = kwargs["policy"]

        self.use_cut_detector = False
        if "cut_detector" in kwargs:
            self.use_cut_detector = True
            self.cut_detector = kwargs["cut_detector"]

        self.curr_state = StateRecorder()
        self.prev_state = StateRecorder()

        self.prev_node = NodeRecorder()
        self.curr_node = NodeRecorder()
        self.last_cut_node = NodeRecorder()
        self.root_node = NodeRecorder()
        self.cut_nodes = set()

        self.total_cuts = 0
        self.total_reward = 0
        self.root_cuts: Iterator[List[Any]] = kwargs["root_cuts"] if "root_cuts" in kwargs else None

        self.actions = {0: 0, 1: 0}
        self.prev_depth = 0
        self.max_obj_improvement = TOLERANCE
        self.max_solution_dist = TOLERANCE
        self.max_nb_cuts = TOLERANCE

        self.frequent = kwargs["frequent"] if "frequent" in kwargs else 1
        self.mdp_terminate = {"terminate": False}


class HeuristicRainbowUserCallback(RainbowUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        self.processed_nodes = self.get_num_nodes()
        node_data = deepcopy(self.get_node_data())

        if node_data is None:
            if self.processed_nodes == 0:
                node_data = {"curr_state": StateRecorder(), "parent_node": NodeRecorder(),
                             "curr_node": NodeRecorder(),
                             "last_cut_node": NodeRecorder(), "nb_generating_cuts": 0}
            elif self.processed_nodes != 0:
                raise Exception(f"Node data is None at the non-root node {self.processed_nodes}")

        # Create the state and node recorders
        self.prev_state, self.last_cut_node = node_data["curr_state"], node_data["last_cut_node"]
        self.nb_generating_cuts = node_data["nb_generating_cuts"]
        self.curr_state, self.curr_node = self.initialize_state_and_node_recorders(node_data)

        if self.prev_state.node_id == self.curr_state.node_id and self.prev_state.obj is not None:
            self.curr_node.obj_improvements.append(abs(self.prev_state.obj - self.curr_state.obj))

        self.curr_node.nb_visited += 1

        solution = np.asarray(self.get_values())
        self.curr_state.solution = coo_matrix(solution)

        if self.processed_nodes == 0:
            self.solve_root_node(solution)
            return

        if self.is_terminal_state():
            self.mdp_terminate["terminate"] = True

        if not self.mdp_terminate["terminate"]:
            solution_graph = self.separator.create_general_support_graph(solution)
            state_representation = self.state_extractor.get_state_representation(self, solution, solution_graph)

            msg = [
                f"Node {self.prev_state.node_id} (t = {self.prev_state.id})",
                f"add {self.prev_state.nb_cuts} user cuts, gap {self.prev_state.gap:.2f}",
                f"total cuts {self.total_cuts}, actions {self.actions}"
            ]
            print(", ".join(msg))

            data = ts.data.batch.Batch({"obs": ts.data.batch.Batch([state_representation]), "info": {}})
            action = self.policy(data)["act"][0]

            if self.curr_node.nb_cuts > 0:
                self.last_cut_node = self.curr_node
                self.cut_nodes.add(self.curr_node.id)

            # Perform the action and update the node data
            self.curr_state.time = time()
            self.curr_state.nb_cuts = -1
            self.actions[action] += 1
            if action == 1:
                self.curr_node.last_cut_distances = []
                support_graph = self.separator.create_support_graph(solution)
                cuts = self.separator.get_user_cuts(support_graph)
                for vars, coefs, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
                    cut_dist = distance_solution_cuts(solution, vars, coefs, rhs)
                    self.curr_node.cut_distances.append(cut_dist)
                    self.curr_node.last_cut_distances.append(cut_dist)

                self.curr_state.nb_cuts = len(cuts)
                self.curr_node.nb_cuts += len(cuts)
                self.total_cuts += self.curr_state.nb_cuts
                self.nb_generating_cuts += 1

            self.prev_depth = self.curr_state.depth
            self.set_node_data({"curr_state": self.curr_state, "parent_node": node_data["parent_node"],
                                "last_cut_node": self.last_cut_node, "curr_node": self.curr_node,
                                "nb_generating_cuts": self.nb_generating_cuts})
        else:
            if self.processed_nodes % self.frequent != 0:
                return

            s = time()
            self.actions[1] += 1
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
            self.total_cuts += len(cuts)
            msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
            )
            if self.logger is not None:
                self.logger.write(msg)
            else:
                print(msg, end="")

    def is_terminal_state(self) -> bool:
        if self.prev_depth > self.curr_state.depth:
            return True

        return False


class CombineHeuristicRainbowUserCallback(RainbowUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        node_data = deepcopy(self.get_node_data())

        if node_data is None:
            if self.processed_nodes == 0:
                node_data = {"curr_state": StateRecorder(), "parent_node": NodeRecorder(),
                             "curr_node": NodeRecorder(),
                             "last_cut_node": NodeRecorder(), "nb_generating_cuts": 0}
            elif self.processed_nodes != 0:
                raise Exception(f"Node data is None at the non-root node {self.processed_nodes}")

        # Create the state and node recorders
        self.prev_state, self.last_cut_node = node_data["curr_state"], node_data["last_cut_node"]
        self.nb_generating_cuts = node_data["nb_generating_cuts"]
        self.curr_state, self.curr_node = self.initialize_state_and_node_recorders(node_data)

        if self.prev_state.node_id == self.curr_state.node_id and self.prev_state.obj is not None:
            self.curr_node.obj_improvements.append(abs(self.prev_state.obj - self.curr_state.obj))

        self.curr_node.nb_visited += 1

        solution = np.asarray(self.get_values())
        self.curr_state.solution = coo_matrix(solution)

        if self.processed_nodes == 0:
            self.solve_root_node(solution)
            return

        if self.is_terminal_state():
            self.mdp_terminate["terminate"] = True

        if not self.mdp_terminate["terminate"] or self.processed_nodes % self.frequent == 0:
            solution_graph = self.separator.create_general_support_graph(solution)
            state_representation = self.state_extractor.get_state_representation(self, solution, solution_graph)

            msg = [
                f"Node {self.prev_state.node_id} (t = {self.prev_state.id})",
                f"add {self.prev_state.nb_cuts} user cuts, gap {self.prev_state.gap:.2f}",
                f"total cuts {self.total_cuts}, actions {self.actions}"
            ]
            self.logger.write(", ".join(msg) + "\n")

            data = ts.data.batch.Batch({"obs": ts.data.batch.Batch([state_representation]), "info": {}})
            action = self.policy(data)["act"][0]

            if self.curr_node.nb_cuts > 0:
                self.last_cut_node = self.curr_node
                self.cut_nodes.add(self.curr_node.id)

            # Perform the action and update the node data
            self.curr_state.time = time()
            self.curr_state.nb_cuts = -1
            self.actions[action] += 1
            if action == 1:
                self.curr_node.last_cut_distances = []
                support_graph = self.separator.create_support_graph(solution)
                cuts = self.separator.get_user_cuts(support_graph)
                for vars, coefs, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
                    cut_dist = distance_solution_cuts(solution, vars, coefs, rhs)
                    self.curr_node.cut_distances.append(cut_dist)
                    self.curr_node.last_cut_distances.append(cut_dist)

                self.curr_state.nb_cuts = len(cuts)
                self.curr_node.nb_cuts += len(cuts)
                self.total_cuts += self.curr_state.nb_cuts
                self.nb_generating_cuts += 1

            self.prev_depth = self.curr_state.depth
            self.set_node_data({"curr_state": self.curr_state, "parent_node": node_data["parent_node"],
                                "last_cut_node": self.last_cut_node, "curr_node": self.curr_node,
                                "nb_generating_cuts": self.nb_generating_cuts})

    def is_terminal_state(self) -> bool:
        if self.prev_depth > self.curr_state.depth:
            return True

        return False
