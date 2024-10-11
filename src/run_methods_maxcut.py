import _pickle
import multiprocessing
import os.path
import socket
from glob import glob
import yaml

import typer

import standard_solve
import rainbow_solve

app = typer.Typer()


@app.command()
def run_baselines(data_root: str, nb_cpus: int):
    frequents = [1, 4, 8]
    problem_type = "maxcut"
    cut_type = "cycle"

    instance_paths = glob(data_root + f"/*.{problem_type}")
    data_folder = "/".join(data_root.split("/")[2:])
    inputs = []
    for f in frequents:
        result_root = f"../results/{data_folder}/{socket.gethostname()}/{f}"
        try:
            if not os.path.isdir(result_root):
                os.makedirs(result_root)
        except:
            print("Folder exist")

        result_path = os.path.join(result_root, "evaluation.csv")
        log_folder = os.path.join(result_root, "logs")
        try:
            if not os.path.isdir(log_folder):
                os.makedirs(log_folder)
        except:
            print("Folder exist")

        for path in instance_paths:
            log_path = os.path.join(log_folder, "{}.log".format(path.split("/")[-1][:-7]))

            if os.path.isfile(log_path):
                continue

            inputs.append((
                problem_type, cut_type, path, True, "BaseUserCallback", None, f, 0.01, result_path, False, log_path,
                3600, None, None, ",", False
            ))

    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(standard_solve.solve, inputs)

    pool.close()
    pool.join()


@app.command()
def run_rainbow(data_root: str, nb_cpus: int, model_path: str, model_name: str = "best_model.pth",
                             user_callback_type: str = "RainbowUserCallback", frequent: int = 1,
                             terminal_gap: float = 0.01, time_limit: int = 3600, dfs_search: bool = True):
    problem_type = "maxcut"
    cut_type = "cycle"

    instance_paths = glob(data_root + f"/*.{problem_type}")
    data_folder = "/".join(data_root.split("/")[2:])
    inputs = []

    result_root = f"../results/{data_folder}/{socket.gethostname()}/{user_callback_type}_{frequent}"
    try:
        if not os.path.isdir(result_root):
            os.makedirs(result_root)
    except:
        print("Folder exist")

    result_path = os.path.join(result_root, "evaluation.csv")
    log_folder = os.path.join(result_root, "logs")
    try:
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)
    except:
        print("Folder exist")

    for path in instance_paths:
        log_path = os.path.join(log_folder, "{}.log".format(path.split("/")[-1][:-7]))

        if os.path.isfile(log_path):
            continue

        inputs.append((
            problem_type,
            cut_type,
            path,
            model_path,
            model_name,
            user_callback_type,
            frequent,
            terminal_gap,
            result_path,
            False,
            log_path,
            time_limit,
            None,
            None,
            ",",
            dfs_search,
        ))

    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(rainbow_solve.rainbow_solve, inputs)

    pool.close()
    pool.join()


@app.command()
def run_skip_factor(data_root: str, nb_cpus: int):
    problem_type = "maxcut"
    cut_type = "cycle"

    instance_paths = glob(data_root + f"/*.{problem_type}")
    data_folder = "/".join(data_root.split("/")[2:])
    inputs = []
    result_root = f"../results/{data_folder}/{socket.gethostname()}/skip_factor"

    try:
        if not os.path.isdir(result_root):
            os.makedirs(result_root)
    except:
        print("Folder exist")

    result_path = os.path.join(result_root, "evaluation.csv")
    log_folder = os.path.join(result_root, "models")
    try:
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)
    except:
        print("Folder exist")

    for path in instance_paths:
        log_path = os.path.join(log_folder, "{}.log".format(path.split("/")[-1][:-7]))

        if os.path.isfile(log_path):
            continue

        inputs.append((
            problem_type, cut_type, path, True, "SkipFactorUserCallback", None, 1, 0.01, result_path, False, log_path,
            3600, None, None, ",", False
        ))

    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(standard_solve.solve, inputs)

    pool.close()
    pool.join()


if __name__ == "__main__":
    app()
