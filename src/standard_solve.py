import _pickle
import os.path
import socket
from time import time
from typing import *

import typer

from problems.problem_name import PROBLEM_NAME
from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.callback_name import CALLBACK_NAME
from solvers.callbacks.base import RecordRootCutCallback
from utils import ResultWriter

app = typer.Typer()


@app.command()
def solve(problem_type: str, cut_type: str, instance_path: str, use_callback: int = True,
          user_callback_type: str = "BaseUserCallback", user_cb_kwargs=None, frequent: int = 1,
          terminal_gap: float = 0.01, result_path: str = "", display_log: bool = True, log_path="",
          time_limit: int = 3600, kws: List[str] = None, external_kws: List[str] = None, separator: str = ",",
          dfs_search: bool = False):
    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path,
                                       time_limit=time_limit)

    if dfs_search:
        solver.parameters.mip.strategy.nodeselect.set(0)
    user_callback = None
    user_cb_kwargs = {} if user_cb_kwargs is None else user_cb_kwargs
    root_cut_path = os.path.join(os.path.dirname(instance_path), "root_cuts", socket.gethostname(),
                                 f"{prob.graph.name}.pkl")
    if os.path.isfile(root_cut_path):
        print("Load root cuts from {}".format(root_cut_path))
        with open(root_cut_path, "rb") as file:
            root_cuts = iter(_pickle.load(file))
        user_cb_kwargs["root_cuts"] = root_cuts
    if use_callback:
        user_callback = solver.register_callback(CALLBACK_NAME[user_callback_type])
        user_callback.set_attribute(solver.separator, terminal_gap=terminal_gap, frequent=frequent,
                                    logger=solver.logger, **user_cb_kwargs)

    s = time()
    solver.solve()
    t = time() - s

    if result_path != "":
        if kws is None or len(kws) == 0:
            kws = ["instanceName", "gap", "nNodes", "nCuts", "nSepa"]

        if external_kws is None or len(external_kws) == 0:
            external_kws = ["time", "frequent"]

        result_writer = ResultWriter(path=result_path, solver=solver, callback=user_callback, kws=kws,
                                     external_kws=external_kws, separator=separator)
        result_writer.write(external_kws={"time": t, "frequent": frequent})


@app.command()
def solve_random(problem_type: str, cut_type: str, instance_path: str, use_callback: int = True, frequent: int = 1,
                 terminal_gap: float = 0.01, result_path: str = "", display_log: int = 1, log_path=""):
    print("Display log", display_log)
    prob = PROBLEM_NAME[problem_type](instance_path)
    if display_log == 0:
        display_log = False
    else:
        display_log = True
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path, time_limit=10800)
    solver.parameters.mip.strategy.nodeselect.set(0)

    user_callback = None
    if use_callback:
        user_callback = solver.register_callback(CALLBACK_NAME["RandomUserCallback"])
        user_callback.set_attribute(solver.separator, terminal_gap=terminal_gap, frequent=frequent,
                                    logger=solver.logger)

    s = time()
    solver.solve()
    t = time() - s

    mode = 1

    if result_path != "":
        if use_callback:
            if not os.path.isfile(result_path):
                with open(result_path, "w") as file:
                    file.write("name,mode,gap,time,total_nodes,total_cuts,action0,action1\n")
            with open(result_path, "a") as file:
                file.write("{},{},{:0.2f},{:0.3f},{},{},{},{}\n".format(prob.graph.graph["name"], mode,
                                                                        solver.solution.MIP.get_mip_relative_gap() * 100,
                                                                        t,
                                                                        user_callback.processed_nodes,
                                                                        user_callback.total_cuts,
                                                                        user_callback.actions[0],
                                                                        user_callback.actions[1]))
        else:
            if not os.path.isfile(result_path):
                with open(result_path, "w") as file:
                    file.write("name,time\n")
            with open(result_path, "a") as file:
                file.write("{},{:0.3f}\n".format(prob.graph.graph["name"], t))


@app.command()
def solve_record_root_cuts(problem_type: str, cut_type: str, instance_path: str, frequent: int = 1,
                           terminal_gap: float = 0.01, display_log=False):
    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log)
    user_callback = solver.register_callback(RecordRootCutCallback)
    user_callback.set_attribute(solver.separator, terminal_gap=terminal_gap, frequent=frequent)

    solver.solve()

    data_root = "/".join(instance_path.split("/")[:-2])
    folder = instance_path.split("/")[-2]
    host_name = socket.gethostname()
    if not os.path.isdir(os.path.join(data_root, folder, "root_cuts", host_name)):
        os.makedirs(os.path.join(data_root, folder, "root_cuts", host_name))
    root_cut_path = os.path.join(data_root, folder, "root_cuts", host_name, "{}.pkl".format(prob.graph.name))
    with open(root_cut_path, "wb") as file:
        _pickle.dump(user_callback.root_cuts, file)


if __name__ == "__main__":
    app()
