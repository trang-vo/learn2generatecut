import random
from copy import deepcopy
from time import time
from typing import *

import numpy as np
from torch.multiprocessing.queue import Queue
from scipy.sparse import coo_matrix

from utils import distance_solution_cuts
from .state_extractor.base import BaseStateExtractor
from .common import StateRecorder, NodeRecorder
from ..cplex_api import cplex, BranchCallback, NodeCallback
from .base import BaseUserCallback
from constant import TOLERANCE


class RewardCalculator:
    def __init__(self, reward_type: str, *args, **kwargs):
        self.reward_type = reward_type

    def get_reward(self, callback, **kwargs) -> float:
        reward = 0
        if self.reward_type == "time":
            reward = callback.prev_state.time - time() if callback.prev_state.time is not None else 0
        elif self.reward_type == "time_with_penalty":
            reward = callback.prev_state.time - time() if callback.prev_state.time is not None else 0
            if callback.prev_state.nb_cuts == 0:
                reward *= 2
        elif self.reward_type == "reward_shaping":
            action_cost = -0.01
            reinforce_cuts = 0
            if callback.prev_cuts > 0:
                reinforce_cuts = callback.prev_cuts * 0.01
            elif callback.prev_cuts == 0:
                reinforce_cuts = -0.1
            reward = action_cost + reinforce_cuts
        elif self.reward_type == "time_reward_shaping":
            if callback.prev_time is not None:
                reward = -(time() - callback.prev_time)
            else:
                reward = 0

            bonus = 0

            if callback.prev_obj is not None:
                print(callback.has_incumbent(), callback.get_cutoff())
                diff_obj = (-callback.prev_obj + callback.get_objective_value()) / callback.get_best_objective_value()
                print("Bonus from diff obj", diff_obj)
                bonus += diff_obj

            if callback.prev_gap is not None:
                diff_gap = (callback.prev_gap - kwargs["gap"]) * 100
                print("Bonus from diff gap", diff_gap)
                bonus += diff_gap

            reward += bonus
        elif self.reward_type == "gap_reward_shaping":
            # assert "gap" in kwargs, "Provide gap to compute reward"
            reward = (callback.prev_state.gap - callback.curr_state.gap) * 100
            if abs(reward) < TOLERANCE:
                reward = -0.01
                if callback.prev_state.nb_cuts == 0:
                    reward = -0.1
            rw_shaping = 0
            if callback.prev_state.nb_cuts > 0:
                rw_shaping += callback.prev_state.nb_cuts * 0.01
            if callback.prev_state.obj is not None:
                rw_shaping += abs(callback.prev_state.obj - callback.curr_state.obj) / callback.curr_state.obj
            reward += rw_shaping
        elif self.reward_type == "time_distance":
            if callback.prev_time is not None:
                reward = -(time() - callback.prev_time)
            else:
                reward = 0

            rw_shaping = 0
            bonus_dist = 0
            for vars, coefs, sense, rhs in callback.prev_list_cuts:
                dist = distance_solution_cuts(callback.optimal_solution, vars, coefs, rhs)
                bonus_dist += np.sign(0.1 - dist) * ((0.1 - dist) ** 2) * 10
            rw_shaping += bonus_dist
            print("Bonus from distance", bonus_dist)

            reward += rw_shaping
        elif self.reward_type == "relative_time_distance":
            if callback.prev_time is not None:
                reward = -(time() - callback.prev_time)
            else:
                reward = 0

            if len(callback.prev_list_cuts) > 0:
                time_find_a_cut = reward / len(callback.prev_list_cuts)
                distances = []
                for vars, coefs, sense, rhs in callback.prev_list_cuts:
                    dist = distance_solution_cuts(callback.optimal_solution, vars, coefs, rhs)
                    distances.append(dist)

                distances = np.asarray(distances)
                cut_cost = np.sum(np.where(distances > TOLERANCE, 1, 0)) * time_find_a_cut - np.mean(distances)
                reward = cut_cost / len(callback.prev_list_cuts)

        return reward


class EnvUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
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

        state, reward, done, info = None, 0, False, {"total_reward": self.total_reward,
                                                     "node_id": self.curr_node.id,
                                                     "total_cuts": self.total_cuts}
        done, terminal_reward = self.is_terminal_state()

        # Determine the initial state
        if not self.has_started_MDP:
            if self.is_initial_state(node_data):
                print(f"The MDP has started at the node {self.processed_nodes}")
                self.has_started_MDP = True

        self.curr_node.nb_visited += 1

        solution = np.asarray(self.get_values())
        self.curr_state.solution = coo_matrix(solution)

        if self.processed_nodes == 0:
            self.solve_root_node(solution)
            return

        # Determine the action. If the MDP has started, the action is determined by the policy.
        # Otherwise, the action is determined by the random policy.
        support_graph = None
        if self.has_started_MDP:
            solution_graph = self.separator.create_general_support_graph(solution)
            state_representation = self.state_extractor.get_state_representation(self, solution, solution_graph)
            if done:
                self.state_queue.put((state_representation, terminal_reward, done, info))
                self.abort()
                return

            reward = self.reward_calculator.get_reward(self)
            self.total_reward += reward
            msg = [
                f"Node {self.prev_state.node_id} (t = {self.prev_state.id})",
                f"add {self.prev_state.nb_cuts} user cuts, gap {self.prev_state.gap:.2f}, reward {reward:.2f}",
                f"total cuts {self.total_cuts}, actions {self.actions}, total reward {self.total_reward:.2f}"
            ]
            print(", ".join(msg))

            self.state_queue.put((state_representation, reward, done, info))
            action = self.action_queue.get()
        else:
            action = np.random.randint(0, 2)
            self.curr_state.id = 0

        if self.curr_node.nb_cuts > 0:
            self.last_cut_node = self.curr_node
            self.cut_nodes.add(self.curr_node.id)

        # Perform the action and update the node data
        self.curr_state.time = time()
        self.curr_state.nb_cuts = -1
        self.actions[action] += 1
        if action == 1:
            self.curr_node.last_cut_distances = []
            if support_graph is None:
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
            if self.has_started_MDP:
                self.nb_generating_cuts += 1

        self.prev_depth = self.curr_state.depth
        self.set_node_data({"curr_state": self.curr_state, "parent_node": node_data["parent_node"],
                            "last_cut_node": self.last_cut_node, "curr_node": self.curr_node,
                            "nb_generating_cuts": self.nb_generating_cuts})

    # Initialize the current state and node recorders
    def initialize_state_and_node_recorders(self, node_data: Dict[str, Any]) -> Tuple[StateRecorder, NodeRecorder]:
        curr_state = StateRecorder()
        curr_state.depth = self.get_current_node_depth()
        curr_state.obj = self.get_objective_value()
        curr_state.gap = min(self.get_MIP_relative_gap(), 1)
        curr_state.id = node_data["curr_state"].id + 1 if node_data["curr_state"].id is not None else 0
        curr_state.node_id = self.processed_nodes

        curr_node = node_data["curr_node"]
        if self.processed_nodes != curr_node.id:
            curr_node = NodeRecorder()
            curr_node.id = self.processed_nodes
            curr_node.depth = self.get_current_node_depth()

        return curr_state, curr_node

    # Solve the root node
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

    # Verify whether the current state is the terminal state
    def is_terminal_state(self) -> Tuple[bool, float]:
        if self.mdp_type == "simplified":
            if self.has_started_MDP and self.prev_depth > self.curr_state.depth:
                reward = - self.curr_state.gap * 100
                print("The simplified MDP terminates")
                return True, reward
        elif self.mdp_type == "original":
            if self.curr_state.gap < 0.01:
                reward = (self.prev_state.gap - self.curr_state.gap) * 100
                print("The original MDP terminates")
                return True, reward

        return False, 0

    # Determine whether the current node is the initial state
    def is_initial_state(self, node_data: Dict[str, Any]) -> bool:
        """
        Determine whether the current node is the initial state
        A node can be considered as the initial state if it satisfies the following conditions:
        + The node has yet to be processed, namely that the number of visiting times is 0
        + The node meets one of the following conditions:
            + The node's parent node is the root node
            + The node's depth is less than or equal to the previous depth
        + The probability of selecting the node is greater than or equal to the given probability
        """
        if random.random() > self.initial_probability:
            return False

        if self.curr_node.id == 0:
            return False

        if self.curr_node.nb_visited > 0:
            return False

        if node_data["parent_node"].id != 0 and self.curr_node.depth > self.prev_depth - 5:
            return False

        return True

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: BaseStateExtractor = kwargs["state_extractor"]
        self.state_queue: Queue = kwargs["state_queue"]
        self.action_queue: Queue = kwargs["action_queue"]
        self.env_mode: str = kwargs["env_mode"]
        self.config: Dict[str, Any] = kwargs["config"]
        self.mdp_type: str = kwargs.get("mdp_type", "original")
        self.reward_calculator = RewardCalculator(reward_type=self.config["reward_type"])
        self.has_started_MDP = False

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

        self.initial_probability = kwargs["initial_probability"] if "initial_probability" in kwargs else 1.0


class RecordBranchCallback(BranchCallback):
    def __call__(self, *args, **kwargs):
        parent_node_data = self.get_node_data()
        if parent_node_data is None:
            parent_node_data = {"depth": 0}

        for i in range(self.get_num_branches()):
            self.make_cplex_branch(i, node_data={"depth": parent_node_data["depth"] + 1})


class DFSRandomNodeCallback(NodeCallback):
    def __call__(self, *args, **kwargs):
        if not self.mdp_terminate["terminate"]:
            node_depth = {node: self.get_node_data(node)["curr_node"].depth for node in range(self.get_num_remaining_nodes())}
            max_depth = max(node_depth.values())
            nodes_with_max_depth = [key for key, value in node_depth.items() if value == max_depth]
            self.select_node(nodes_with_max_depth[0])

    def set_attribute(self, mdp_terminate: dict, *args, **kwargs):
        self.mdp_terminate = mdp_terminate


class RandomStartBranchCallback(BranchCallback):
    def __call__(self, *args, **kwargs):
        parent_node_data = self.get_node_data()

        for i in range(self.get_num_branches()):
            node_data = deepcopy(parent_node_data)
            node_data["parent_node"] = parent_node_data["curr_node"]
            self.make_cplex_branch(i, node_data=node_data)
