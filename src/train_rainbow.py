import argparse
import datetime
import json
import os
import socket

import numpy as np
import torch.optim
import yaml
from lescode.config import load_config
from lescode.namespace import asdict

from environments.subproc_env import make_env, MultiSubprocVectorEnv
from agents.ts_dqn import TsDQN
import utils
from agents.feature_extractors.feature_extractor_name import FEATURE_EXTRACTOR_NAME
from agents.feature_extractors.feature_extractors import TsFeatureExtractor


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_type', type=str)
    parser.add_argument('cut_type', type=str)
    parser.add_argument('env_name', type=str)
    parser.add_argument('seed', type=int)
    args = parser.parse_known_args()[0]
    return args


def train_agent(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    entry_config = load_config(name="entry", path="configs/{}.yaml".format(args.cut_type)).detail
    env_config = asdict(load_config(name="env", path=entry_config.env_config).detail)
    agent_config = asdict(load_config(name="agent", path=entry_config.agent_config).detail)

    logdir = "../logs"
    now = datetime.datetime.now()
    t = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)
    host_name = socket.gethostname()
    folder = "{}{}_TianshouDQN_{}_{}".format(args.cut_type.capitalize(), args.env_name, t, host_name)
    if not os.path.isdir(os.path.join(logdir, folder)):
        os.makedirs(os.path.join(logdir, folder))
    log_path = os.path.join(logdir, folder) + "/"

    utils.copy_file_to_folder(entry_config.env_config, log_path, new_file_name=f"env_{args.cut_type}.yaml")
    with open(os.path.join(log_path, "run_config.json"), "w") as f:
        json.dump(vars(args), f)

    n_cpu = agent_config["train"]["n_cpu"]
    env_fns = [make_env(args.env_name, args.problem_type, args.cut_type, mode="train", config=env_config)
               for _ in range(n_cpu)]
    train_envs = MultiSubprocVectorEnv(env_fns)
    train_envs.seed(2)
    eval_env = MultiSubprocVectorEnv(
        [make_env(args.env_name, args.problem_type, args.cut_type, mode="eval",
                  result_path=log_path + "/result_eval.csv",
                  config=env_config)
         for _ in range(1)])

    device = agent_config["device"] if torch.cuda.is_available() else "cpu"
    sup_feature_extractor = None
    ori_feature_extractor = None
    statistic_extractor = None

    if agent_config["state_components"].get("statistic", None) is not None:
        statistic_config = asdict(
            load_config(name="statistic", path=agent_config["state_components"]["statistic"]).detail)
        utils.copy_file_to_folder(agent_config["state_components"]["statistic"], log_path,
                                  new_file_name=f"statistic.yaml")
        statistic_extractor = FEATURE_EXTRACTOR_NAME[statistic_config["model_type"]](**statistic_config["params"],
                                                                                     device=device)
        agent_config["state_components"]["statistic"] = "statistic.yaml"


    if agent_config["state_components"].get("solution", None) is not None:
        solution_config = asdict(
            load_config(name="solution", path=agent_config["state_components"]["solution"]).detail)
        utils.copy_file_to_folder(agent_config["state_components"]["solution"], log_path,
                                  new_file_name=f"solution.yaml")
        sup_feature_extractor = FEATURE_EXTRACTOR_NAME[solution_config["model_type"]](**solution_config["params"],
                                                                                      device=device)
        agent_config["state_components"]["solution"] = "solution.yaml"

    if agent_config["state_components"].get("problem", None) is not None:
        problem_config = asdict(
            load_config(name="problem", path=agent_config["state_components"]["problem"]).detail)
        utils.copy_file_to_folder(agent_config["state_components"]["problem"], log_path,
                                  new_file_name=f"problem.yaml")
        ori_feature_extractor = FEATURE_EXTRACTOR_NAME[problem_config["model_type"]](**problem_config["params"],
                                                                                      device=device)
        agent_config["state_components"]["problem"] = "problem.yaml"

    with open(os.path.join(log_path, f"agent_{args.cut_type}.yaml"), 'w') as outfile:
        yaml.dump(agent_config, outfile, default_flow_style=False)

    net = TsFeatureExtractor(action_shape=agent_config["action_shape"], solution_encoder=sup_feature_extractor,
                             problem_encoder=ori_feature_extractor, statistic_encoder=statistic_extractor,
                             use_atoms=agent_config["policy"]["use_atoms"],
                             num_atoms=agent_config["policy"]["num_atoms"], device=device)

    agent = TsDQN(net=net, optimizer_config=agent_config["optimizer"], policy_config=agent_config["policy"], )
    results = agent.train(train_env=train_envs, eval_env=eval_env, log_path=log_path,
                          learn_config=agent_config["train"])
    with open(os.path.join(log_path, "results_training.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_agent(args)
