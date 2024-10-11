from typing import Dict, Any

import _pickle
import os
import socket
from time import time
from typing import *

import torch
import typer
from lescode.config import load_config
from lescode.namespace import asdict
from torch import nn as nn

from agents.ts_dqn import TsDQN
from cut_detectors.subtour import SubtourDetector
from agents.feature_extractors.feature_extractor_name import FEATURE_EXTRACTOR_NAME
from agents.feature_extractors.feature_extractors import TsFeatureExtractor
from problems.problem_name import PROBLEM_NAME
from solvers.callbacks.callback_name import CALLBACK_NAME
from solvers.callbacks.env_callback import RandomStartBranchCallback
from solvers.callbacks.state_extractor.state_extractor_name import STATE_EXTRACTOR_NAME
from solvers.solver_name import SOLVER_NAME
from agents.policy_net.policy_net_name import POLICY_NET_NAME
from utils import ResultWriter

app = typer.Typer()


def create_feature_extractor(agent_config_path: str, agent_config: Dict[str, Any]):
    device = agent_config["device"] if torch.cuda.is_available() else "cpu"
    sup_feature_extractor = None
    ori_feature_extractor = None
    statistic_extractor = None

    if agent_config["state_components"].get("statistic", None) is not None:
        path = os.path.join(agent_config_path, agent_config["state_components"]["statistic"])
        statistic_config = asdict(
            load_config(name="statistic", path=path).detail)
        statistic_extractor = FEATURE_EXTRACTOR_NAME[statistic_config["model_type"]](**statistic_config["params"],
                                                                                     device=device)

    if agent_config["state_components"].get("solution", None) is not None:
        path = os.path.join(agent_config_path, agent_config["state_components"]["solution"])
        solution_config = asdict(
            load_config(name="solution", path=path).detail)
        sup_feature_extractor = FEATURE_EXTRACTOR_NAME[solution_config["model_type"]](**solution_config["params"],
                                                                                      device=device)

    if agent_config["state_components"].get("problem", None) is not None:
        path = os.path.join(agent_config_path, agent_config["state_components"]["problem"])
        problem_config = asdict(
            load_config(name="problem", path=path).detail)
        ori_feature_extractor = FEATURE_EXTRACTOR_NAME[problem_config["model_type"]](**problem_config["params"],
                                                                                     device=device)

    feature_extractor = TsFeatureExtractor(action_shape=agent_config["action_shape"],
                                           solution_encoder=sup_feature_extractor,
                                           problem_encoder=ori_feature_extractor, statistic_encoder=statistic_extractor,
                                           use_atoms=agent_config["policy"].get("use_atoms", True),
                                           num_atoms=agent_config["policy"].get("num_atoms", 51), device=device)

    return feature_extractor


def create_policy_net(feature_extractor: nn.Module, agent_config: Dict[str, Any]):
    policy_config = agent_config["policy_net"]
    policy_net = POLICY_NET_NAME[policy_config["model_type"]](feature_extractor=feature_extractor,
                                                              action_shape=agent_config["action_shape"],
                                                              num_atoms=agent_config["policy"].get("num_atoms", 51),
                                                              device=agent_config["device"],
                                                              **policy_config)
    return policy_net


@app.command()
def rainbow_solve(
        problem_type: str,
        cut_type: str,
        instance_path: str,
        model_path: str,
        model_name: str = None,
        user_callback_type: str = "RainbowUserCallback",
        frequent: int = 1,
        terminal_gap: float = 0.01,
        result_path: str = "",
        display_log: bool = True,
        log_path="",
        time_limit: int = 3600,
        kws: List[str] = None,
        external_kws: List[str] = None,
        separator: str = ",",
        dfs_search: bool = True,
        use_cut_detector: bool = False,
        cut_detector_path: str = "",
        device: str = "cuda",
):
    env_config = asdict(load_config(name="env", path=os.path.join(model_path, f"env_{cut_type}.yaml")).detail)
    agent_config = asdict(load_config(name="agent", path=os.path.join(model_path, f"agent_{cut_type}.yaml")).detail)

    # Create the feature extractor
    feature_extractor = create_feature_extractor(model_path, agent_config)

    # Create the policy net
    policy_net = create_policy_net(feature_extractor, agent_config)
    print(type(policy_net))

    # Create the agent
    agent = TsDQN(net=policy_net, optimizer_config=agent_config["optimizer"], policy_config=agent_config["policy"],
                  device=device)
    if model_name is None:
        model_name = "best_model.pth"

    model_path = os.path.join(model_path, model_name)
    assert os.path.isfile(model_path), f"{model_path} is invalid dir"

    if model_path != "":
        agent.policy.load_state_dict(torch.load(model_path))
    agent.policy.eval()

    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path,
                                       time_limit=time_limit)

    if dfs_search:
        solver.parameters.mip.strategy.nodeselect.set(0)

    state_extractor = STATE_EXTRACTOR_NAME[cut_type][env_config["state_extractor_class"]](env_config["state_components"],
                                                                                          mdp_type=env_config["mdp_type"],
                                                                                          padding=False)
    state_extractor.initialize_original_graph(prob, solver.edge2idx)

    user_callback = solver.register_callback(CALLBACK_NAME[user_callback_type])
    user_cb_kwargs = {
        "state_extractor": state_extractor,
        "config": env_config["episode_config"],
        "policy": agent.policy,
        "frequent": frequent,
        "terminal_gap": terminal_gap,
        "logger": solver.logger,
    }
    root_cut_path = os.path.join(os.path.dirname(instance_path), "root_cuts", socket.gethostname(),
                                 "{}.pkl".format(prob.graph.name))
    if os.path.isfile(root_cut_path):
        print("Load root cuts from {}".format(root_cut_path))
        with open(root_cut_path, "rb") as file:
            root_cuts = iter(_pickle.load(file))
        user_cb_kwargs["root_cuts"] = root_cuts

    if use_cut_detector:
        cut_detector_config_path = os.path.join(*cut_detector_path.split("/")[:-1], "config.yaml")
        cut_detector_config = load_config("cut_detectors", path=cut_detector_config_path).detail
        cut_detector = SubtourDetector(cut_detector_config.node_dim, cut_detector_config.hidden_size,
                                       cut_detector_config.output_size, cut_detector_config, device)
        cut_detector.load_state_dict(torch.load(cut_detector_path))
        if torch.cuda.is_available():
            cut_detector = cut_detector.to(device)
        user_cb_kwargs["cut_detector"] = cut_detector

    user_callback.set_attribute(solver.separator, **user_cb_kwargs)
    solver.register_callback(RandomStartBranchCallback)

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


if __name__ == "__main__":
    app()



