import json
import os
from time import time
from typing import *
import sys

import torch
from lescode.config import load_config
import typer

from agents.feature_extractors.state_encoder import EvalFeatureExtractor
from problems.problem_name import PROBLEM_NAME
from solvers.callbacks.state_extractor.state_extractor_name import STATE_EXTRACTOR_NAME
from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.callback_name import CALLBACK_NAME
from agents.feature_extractors.feature_extractor_name import FEATURE_EXTRACTOR_NAME
from cut_detectors.subtour import SubtourDetector


def create_features_extractor(extractor_config: Dict[str, Any], env_config: Dict[str, Any], **kwargs):
    if "device" in kwargs:
        device = torch.device(kwargs["device"])
    else:
        device = torch.device("cpu")

    sup_feature_extractor = FEATURE_EXTRACTOR_NAME[
        extractor_config["sup_feature_extractor"]
    ](
        node_dim=env_config["sup_node_dim"],
        edge_dim=env_config["sup_edge_dim"],
        hidden_size=extractor_config["sup_hidden_size"],
        num_layers=extractor_config["sup_num_layers"],
        n_clusters=extractor_config["sup_n_clusters"],
        dropout=extractor_config["sup_dropout"],
        device=device,
    )

    ori_feature_extractor = FEATURE_EXTRACTOR_NAME[
        extractor_config["ori_feature_extractor"]
    ](
        node_dim=env_config["ori_node_dim"],
        edge_dim=env_config["ori_edge_dim"],
        hidden_size=extractor_config["ori_hidden_size"],
        num_layers=extractor_config["ori_num_layers"],
        n_clusters=extractor_config["ori_n_clusters"],
        dropout=extractor_config["ori_dropout"],
        device=device,
    )

    statistic_extractor = FEATURE_EXTRACTOR_NAME[extractor_config["statistic_extractor"]](
        input_size=env_config["statistic_dim"],
        hidden_sizes=extractor_config["statistic_hidden_sizes"],
        output_size=extractor_config["statistic_output_size"],
        device=device,
    )

    return EvalFeatureExtractor(sup_feature_extractor, ori_feature_extractor, statistic_extractor, device=device)


def convert_config(conf: Dict[Any, Any]):
    new_conf = {"env": {
        "space_config": {
            "instance_size": 200,
            "ori_nEdges": 3000,
            "ori_node_dim": conf["CutEnv"]["ori_node_dim"],
            "ori_edge_dim": conf["CutEnv"]["ori_edge_dim"],
            "statistic_dim": conf["CutEnv"]["static_dim"],
            "sup_nNodes": 200,
            "sup_nEdges": 500,
            "sup_node_dim": conf["CutEnv"]["node_dim"],
            "sup_edge_dim": conf["CutEnv"]["edge_dim"],
        },
        "mdp_type": "original",
        "state_extractor_class": "SubtourStateExtractor",
        "state_components": ["solution", "statistic", "problem"],
        "data_folder": "../data/tsp_instances/200",
        "k_nearest_neighbors": 10,
        "episode_config": {
            "limited_one_action": 2000,
            "time_limit": 3000,
            "reward_time_limit": -1000,
            "terminal_gap": 0.01,
            "reward_type": "time"
        },
        "initial_start_distance": 0.05,
        "final_start_distance": 1,
        "start_distance_fraction": 0.5,
        "total_train_steps": 1000000
    }, "extractor": {
        "sup_feature_extractor": "GNNGraphExtractor",
        "sup_hidden_size": 64,
        "sup_num_layers": 2,
        "sup_n_clusters": 2,
        "sup_dropout": 0,
        "ori_feature_extractor": "GINEGraphExtractor",
        "ori_hidden_size": 64,
        "ori_num_layers": 2,
        "ori_n_clusters": 2,
        "ori_dropout": 0,
        "statistic_extractor": "MLPOneLayer",
        "statistic_hidden_sizes": [64],
        "statistic_output_size": 64
    }}

    return new_conf


def load_agent(agent_root: str, agent_folder: str, agent_name: str, config: Dict[str, Any],
               device: torch.device = "cpu"):
    agent_path = os.path.join(agent_root, agent_folder, agent_name)
    if not torch.cuda.is_available():
        model = torch.load(agent_path, map_location=torch.device("cpu"))
    else:
        model = torch.load(agent_path)
    assert model is not None

    features_extractor = create_features_extractor(extractor_config=config["extractor"],
                                                   env_config=config["env"]["space_config"], device=device)
    features_extractor.load_state_dict(model["features_extractor"])
    agent = model["agent"]

    return features_extractor, agent


def solve(problem_type: str, cut_type: str, instance_path: str, use_cut_detector: bool = True,
          cut_detector_path: str = "", use_rl_agent: bool = True, agent_root: str = "", agent_folder: str = "",
          agent_name: str = "", frequent: int = 1, terminal_gap: float = 0.01, result_path: str = "",
          display_log: bool = True, log_path: str = "", device: str = "cpu", time_limit: int = 3600,
          only_root: bool = False) -> None:
    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path,
                                       time_limit=time_limit)

    agent_config_path = os.path.join(agent_root, agent_folder, "config.json")
    with open(agent_config_path, "r") as file:
        agent_config = json.load(file)
        if "env" not in agent_config or "extractor" not in agent_config:
            agent_config = convert_config(agent_config)

    state_extractor = STATE_EXTRACTOR_NAME[cut_type]["default"](["solution", "problem", "statistic"], padding=False)
    state_extractor.initialize_original_graph(prob, solver.edge2idx, k=agent_config["env"]["k_nearest_neighbors"])

    user_cb_kwargs = {
        "frequent": frequent,
        "terminal_gap": terminal_gap,
        "state_extractor": state_extractor,
        "config": agent_config["env"]["episode_config"],
        "logger": solver.logger if solver.logger is not None else sys.stdout,
        "only_root": only_root,
    }

    if device != "cpu" and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    print("Using device", device)

    if use_cut_detector:
        cut_detector_config_path = os.path.join(*cut_detector_path.split("/")[:-1], "config.yaml")
        cut_detector_config = load_config("cut_detectors", path=cut_detector_config_path).detail
        cut_detector = SubtourDetector(cut_detector_config.node_dim, cut_detector_config.hidden_size,
                                       cut_detector_config.output_size, cut_detector_config, device)
        cut_detector.load_state_dict(torch.load(cut_detector_path))
        if torch.cuda.is_available():
            cut_detector = cut_detector.to(device)
        user_cb_kwargs["cut_detector"] = cut_detector

    if use_rl_agent:
        agent_config_path = os.path.join(agent_root, agent_folder, "config.json")
        with open(agent_config_path, "r") as file:
            agent_config = json.load(file)
            if "env" not in agent_config or "extractor" not in agent_config:
                agent_config = convert_config(agent_config)
        features_extractor, agent = load_agent(agent_root, agent_folder, agent_name, config=agent_config, device=device)
        features_extractor = features_extractor.to(device)
        agent = agent.to(device)
        user_cb_kwargs["features_extractor"] = features_extractor
        user_cb_kwargs["policy"] = agent

    user_callback = solver.register_callback(CALLBACK_NAME["MLUserCallback"])
    user_callback.set_attribute(solver.separator, **user_cb_kwargs)

    s = time()
    solver.solve()
    t = time() - s

    if result_path != "":
        if not os.path.isfile(result_path):
            with open(result_path, "w") as file:
                file.write("name,use_cut_detector,use_rl_agent,gap,time,total_nodes,total_cuts,action0,action1\n")
        with open(result_path, "a") as file:
            file.write("{},{},{},{:0.2f},{:0.4f},{},{},{},{}\n".format(prob.graph.graph["name"], use_cut_detector,
                                                                       use_rl_agent,
                                                                       solver.solution.MIP.get_mip_relative_gap() * 100,
                                                                       t,
                                                                       user_callback.processed_nodes,
                                                                       user_callback.total_cuts,
                                                                       user_callback.actions[0],
                                                                       user_callback.actions[1]))


def solve_debug():
    problem_type = "tsp"
    cut_type = "subtour"
    instance_path = "../data/tsplib/kroB200.tsp"
    use_cut_detector = True
    cut_detector_path = "../models/subtour_detector/GNN_mincut.pt"
    use_rl_agent = True
    agent_root = "../logs"
    agent_folder = "subtour_gap"
    agent_name = "best_model.pt"
    frequent = 1
    terminal_gap = 0.01
    result_path = ""
    display_log = True
    log_path = ""

    solve(problem_type, cut_type, instance_path, use_cut_detector, cut_detector_path, use_rl_agent, agent_root,
          agent_folder, agent_name)


if __name__ == "__main__":
    typer.run(solve)