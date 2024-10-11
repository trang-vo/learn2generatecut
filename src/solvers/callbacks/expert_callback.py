from collections import defaultdict
from time import time
from typing import *

import numpy as np

from .separators.base import Separator
from ..cplex_api import cplex, BranchCallback, NodeCallback
from constant import TOLERANCE
from .base import BaseUserCallback


def get_branch_candidate(callback, solution):
    candidate = None
    max_score = -1
    for i in range(len(solution)):
        if abs(solution[i]) > TOLERANCE and abs(1 - solution[i]) > TOLERANCE:
            cost = callback.get_pseudo_costs(i)
            score = max(cost[0], TOLERANCE) * max(cost[1], TOLERANCE)
            if score > max_score:
                candidate = i
                max_score = score

    return candidate


class PseudoCostBranchCallback(BranchCallback):
    def __call__(self, *args, **kwargs):
        curr_node_data = self.get_node_data()
        if curr_node_data is None:
            curr_node_data = []

        # get branch candidate by pseudo cost
        solution = self.get_values()
        candidate = get_branch_candidate(self, solution)

        if candidate is None:
            for i in range(self.get_num_branches()):
                obj, cans = self.get_branch(i)
                self.make_branch(obj, variables=cans, node_data=curr_node_data + [(cans[0], cans[2])])
            print("Branch candidate is None, use the default strategy")
        else:
            obj = self.get_objective_value()
            self.make_branch(obj, variables=[(int(candidate), "U", 0)], node_data=curr_node_data + [(candidate, 0)])
            self.make_branch(obj, variables=[(int(candidate), "L", 1)], node_data=curr_node_data + [(candidate, 1)])


class RecordNodeCallback(NodeCallback):
    def __call__(self, *args, **kwargs):
        active_node_ids = []
        for node in range(self.get_num_remaining_nodes()):
            active_node_ids.append(self.get_node_ID(node))

        for node_id in self.node_log:
            if node_id not in active_node_ids:
                self.node_log[node_id]["explored"] = True

        for node_id in active_node_ids:
            if node_id not in self.node_log:
                node_data = self.get_node_data(node_id)
                if node_data is None:
                    raise ValueError("Node data of {} is None".format(node_id))

                self.node_log[node_id] = {
                    "branch": node_data,
                    "explored": False,
                }

    def set_attribute(self, node_log: Dict[int, Dict[Any, Any]]) -> None:
        self.node_log = node_log


class LPWorker:
    def __init__(self, solver):
        self.cpx = cplex.Cplex(solver)
        vars_idx = [i for i in range(self.cpx.variables.get_num())]
        vars_type = [self.cpx.variables.type.continuous] * len(vars_idx)
        self.cpx.variables.set_types(list(zip(vars_idx, vars_type)))

        self.cpx.set_results_stream(None)
        self.cpx.set_log_stream(None)
        self.cpx.set_warning_stream(None)
        self.cpx.set_error_stream(None)

    def add_cut(self, cuts: List[Tuple[List[int], List[int], str, float]]):
        for vars_, coefs_, sense, rhs in cuts:
            self.cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars_, coefs_)], senses=[sense], rhs=[rhs])

    def solve_with_additional_constraints(self, constraints: List[Tuple[List[int], List[int], str, float]]):
        cpx = cplex.Cplex(self.cpx)
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_error_stream(None)
        for vars_, coefs_, sense, rhs in constraints:
            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars_, coefs_)], senses=[sense], rhs=[rhs])

        cpx.solve()
        solution, obj_value = None, None
        if cpx.solution.get_status() == cpx.solution.status.MIP_optimal:
            solution = cpx.solution.get_values()
            obj_value = cpx.solution.get_objective_value()

        return solution, obj_value


class ExpertUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        if self.processed_nodes % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        if self.mode == "follower":
            action = 0
            if self.processed_nodes in self.expert_policy:
                action = next(self.expert_policy[self.processed_nodes], 0)
            if self.processed_nodes == 0 or action == 1:
                s = time()
                solution = np.asarray(self.get_values())
                support_graph = self.separator.create_support_graph(solution)
                cuts = self.separator.get_user_cuts(support_graph)

                for vars_, coefs_, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars_, coefs_), sense=sense, rhs=rhs)
                self.logger.write(
                    "At node {}, add {} user cuts in {:.4f}s\n".format(self.processed_nodes, len(cuts), time() - s))
            return

        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        if len(cuts) > 0:
            self.portion_cuts["with_cut"] += 1
        else:
            self.portion_cuts["without_cut"] += 1

        if self.processed_nodes == 0:
            for vars_, coefs_, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars_, coefs_), sense=sense, rhs=rhs)
            self.lp_worker.add_cut(cuts)
            return

        if len(cuts) == 0:
            return

        prev_branches = self.get_node_data()
        b_constraints = []
        if prev_branches is not None:
            for var, val in prev_branches:
                b_constraints.append(([var], [1], "E", val))

        c_sol, c_obj_val = self.lp_worker.solve_with_additional_constraints(b_constraints + cuts)

        branch_candidate = get_branch_candidate(self, solution)
        b0_sol, b0_obj_val = self.lp_worker.solve_with_additional_constraints(
            b_constraints + [([branch_candidate], [1], "E", 0)])
        b1_sol, b1_obj_val = self.lp_worker.solve_with_additional_constraints(
            b_constraints + [([branch_candidate], [1], "E", 1)])

        curr_obj = self.get_objective_value()

        if c_obj_val is None:
            c_score = 1e+9
        else:
            c_score = abs(curr_obj - c_obj_val)

        if b0_obj_val is None or b1_obj_val is None:
            b_score = 1e+9
        else:
            b_score = 5 / 6 * min(abs(curr_obj - b0_obj_val), abs(curr_obj - b1_obj_val)) \
                      + 1 / 6 * max(abs(curr_obj - b0_obj_val), abs(curr_obj - b1_obj_val))

        self.logger.write("Cut score: {:.4f}, branch score: {:.4f}\n".format(c_score, b_score))

        if c_score >= b_score:
            self.expert_policy[self.processed_nodes].append(1)
            for vars_, coefs_, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars_, coefs_), sense=sense, rhs=rhs)
            self.logger.write("At node {}, add {} user cuts\n".format(self.processed_nodes, len(cuts)))
            self.lp_worker.add_cut(cuts)
            return
        else:
            self.expert_policy[self.processed_nodes].append(0)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super(ExpertUserCallback, self).set_attribute(separator, *args, **kwargs)
        self.added_cuts = []
        self.expert_policy: Dict[int, List[int]] = defaultdict(list)
        self.mode = kwargs["mode"] if "mode" in kwargs else "expert"
        if self.mode not in ["expert", "follower"]:
            raise ValueError("Mode can be only expert or follower, provided", self.mode)

        if self.mode == "follower":
            if "expert_policy" in kwargs:
                self.expert_policy = kwargs["expert_policy"]
            else:
                raise KeyError("You need to provide an expert policy when using the follower mode")
        else:
            self.lp_worker: LPWorker = kwargs["lp_worker"]
