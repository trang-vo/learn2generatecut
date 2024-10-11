import json
import os.path
from time import time

import solvers.callbacks.expert_callback as ec
from problems.problem_name import PROBLEM_NAME
from solvers.solver_name import SOLVER_NAME


def expert_solve(problem_type: str, cut_type: str, instance_path: str, frequent: int = 1,
                 terminal_gap: float = 0.01, result_path: str = "", display_log: int = 1, log_path: str = "",
                 time_limit: int = 3600):
    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path,
                                       time_limit=time_limit)
    solver.register_callback(ec.PseudoCostBranchCallback)
    node_log = dict()
    node_callback = solver.register_callback(ec.RecordNodeCallback)
    node_callback.set_attribute(node_log=node_log)
    user_callback = solver.register_callback(ec.ExpertUserCallback)
    lp_worker = ec.LPWorker(solver)
    user_callback.set_attribute(separator=solver.separator, lp_worker=lp_worker, logger=solver.logger,
                                frequent=frequent, terminal_gap=terminal_gap)

    s = time()
    solver.solve()
    t = time() - s

    policy_folder = "../results/expert_policies/{}_{}".format(problem_type, cut_type)
    if not os.path.isdir(policy_folder):
        os.mkdir(policy_folder)
    policy_path = os.path.join(policy_folder, "{}.json".format(prob.graph.name))
    if not os.path.isfile(policy_path):
        with open(policy_path, "w") as file:
            json.dump(user_callback.expert_policy, file)

    if result_path != "":
        if not os.path.isfile(result_path):
            with open(result_path, "w") as file:
                file.write("name,gap,time,total_nodes,total_cuts,nSepa,nSepaHaveCut\n")
        with open(result_path, "a") as file:
            file.write("{},{:0.2f},{:0.3f},{},{},{},{}\n".format(prob.graph.graph["name"],
                                                                 solver.solution.MIP.get_mip_relative_gap() * 100,
                                                                 t,
                                                                 user_callback.processed_nodes,
                                                                 user_callback.total_cuts,
                                                                 user_callback.portion_cuts["with_cut"] +
                                                                 user_callback.portion_cuts["without_cut"],
                                                                 user_callback.portion_cuts["with_cut"]))


def follower_solve(problem_type: str, cut_type: str, instance_path: str, frequent: int = 1,
                   terminal_gap: float = 0.01, result_path: str = "", display_log: int = 1, log_path: str = "",
                   time_limit: int = 3600, policy_path: str = ""):
    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path,
                                       time_limit=time_limit)
    solver.register_callback(ec.PseudoCostBranchCallback)
    user_callback = solver.register_callback(ec.ExpertUserCallback)
    lp_worker = ec.LPWorker(solver)
    user_callback.set_attribute(separator=solver.separator, lp_worker=lp_worker, logger=solver.logger,
                                frequent=frequent, terminal_gap=terminal_gap)

    with open(policy_path, "r") as file:
        tmp = json.load(file)
    expert_policy = {int(key): iter(value) for key, value in tmp.items()}
    user_callback = solver.register_callback(ec.ExpertUserCallback)
    user_callback.set_attribute(separator=solver.separator, mode="follower", expert_policy=expert_policy)

    solver.solve()
