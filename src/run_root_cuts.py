"""
Pre-solve the instances and record the cuts added at the root node to save time in the training phase of the cut evaluator.
Rerun this file when changing the server since branch-and-cut decisions are dependent on the server.
"""
import glob
import multiprocessing
import os.path
import socket

import typer

from problems.tsp import TSPProblem
from solvers.tsp import TSPSolver
import solvers.callbacks.expert_callback as ec
from standard_solve import solve_record_root_cuts

app = typer.Typer()


@app.command()
def skip_factor_solve(path: str, frequent: int = 1):
    prob = TSPProblem(path=path)
    solver = TSPSolver(prob, cut_type="subtour")

    solver.register_callback(ec.PseudoCostBranchCallback)
    solver.basic_solve(user_callback="BaseUserCallback", user_cb_kwargs={"frequent": frequent})


@app.command()
def record_root_cuts(problem_type: str, cut_type: str, folder_path, n_cpu: int = 16):
    instance_paths = glob.glob(folder_path + f"/*.{problem_type}")
    host_name = socket.gethostname()
    unsolved_instances = []

    for path in instance_paths:
        instance_name = path.split("/")[-1]
        if not os.path.isfile(os.path.join(folder_path, "root_cuts", host_name, instance_name + ".pkl")):
            print("Find root cuts for instance", path)
            unsolved_instances.append(path)

    with multiprocessing.Pool(n_cpu) as pool:
        pool.starmap(solve_record_root_cuts, [(problem_type, cut_type, path) for path in unsolved_instances])


if __name__ == "__main__":
    app()
