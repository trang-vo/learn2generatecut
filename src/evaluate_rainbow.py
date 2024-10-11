import argparse
import json
import os
import socket
import time

import numpy as np
import typer
from lescode.config import load_config
from lescode.namespace import asdict
import tianshou as ts

from src.rainbow_solve import create_feature_extractor, create_policy_net
from agents.ts_dqn import TsDQN
from environments.env_name import ENV_NAME

app = typer.Typer()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrain_path', type=str)
    parser.add_argument('instance_path', type=str)
    parser.add_argument('--problem_type', type=str)
    parser.add_argument('--cut_type', type=str)
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--frequent', type=int)
    args = parser.parse_known_args()[0]
    return args


def rainbow_explore_one_branch(eval_env, instance_path: str, pretrain_path: str, cut_type: str, model_name: str = None,
                               display_log: bool = True):
    agent_config = asdict(
        load_config(name="agent", path=os.path.join(pretrain_path, f"agent_{cut_type}.yaml")).detail)

    # Create the feature extractor
    feature_extractor = create_feature_extractor(pretrain_path, agent_config)

    # Create the policy net
    policy_net = create_policy_net(feature_extractor, agent_config)

    # Create the agent
    agent = TsDQN(net=policy_net, optimizer_config=agent_config["optimizer"], policy_config=agent_config["policy"], )

    t, reward, node_id, cuts = agent.evaluate(eval_env=eval_env, instance_path=instance_path,
                                              pretrain_path=pretrain_path, model_name=model_name,
                                              display_log=display_log)

    return t, reward, node_id, cuts


def explore_branch(problem_type: str, cut_type: str, env_name: str, pretrain_path: str, instance_path: str,
                   mode: str = None, model_name: str = None, frequent: int = 1, display_log: bool = True):
    env_config = asdict(load_config(name="env", path=os.path.join(pretrain_path, f"env_{cut_type}.yaml")).detail)
    env_config["data_folder"] = "/".join(instance_path.split("/")[:-2])

    if mode == "random":
        result_folder = f'{socket.gethostname()}_{mode}'
    elif mode == "frequent":
        result_folder = f'{socket.gethostname()}_{mode}_{frequent}'
    else:
        result_folder = f'{socket.gethostname()}_{mode}_{pretrain_path.split("/")[-1]}_{model_name}'
    result_root = os.path.join(*instance_path.split("/")[:-1], "evaluation", result_folder)
    print("result_root", result_root)
    try:
        if not os.path.isdir(result_root):
            os.makedirs(result_root)
    except FileExistsError:
        pass
    result_path = os.path.join(result_root, "evaluation.csv")

    eval_env = ENV_NAME[env_name](problem_type=problem_type, cut_type=cut_type, mode="eval",
                                  result_path=result_path, **env_config)
    if mode is None or mode == "rainbow":
        t, reward, node_id, cuts = rainbow_explore_one_branch(eval_env, instance_path, pretrain_path,
                                                              cut_type, model_name, display_log)
    else:
        obs, _ = eval_env.reset(display_log=display_log, instance_path=instance_path)
        data = ts.data.batch.Batch({"obs": np.asarray([np.asarray(obs)]), "info": {}})

        node_id = -1
        frequent = frequent
        terminated = False

        s = time.time()
        while not terminated:
            act = 0
            if mode == "random":
                act = np.random.randint(0, 2)
            elif mode == "frequent":
                if node_id > 0 and node_id % frequent == 0:
                    act = 1
                else:
                    act = 0
            obs, reward, terminated, _, info = eval_env.step(act)
            node_id = info["node_id"]
            cuts = info["total_cuts"]
            data["obs"] = np.asarray([obs])
            data["info"] = info
            data["reward"] = reward
        t = time.time() - s

    return instance_path.split("\n")[-1], t, reward, node_id, cuts


@app.command()
def evaluate_one_instance(instance_path: str, problem_type: str = None, cut_type: str = None, env_name: str = None,
                          pretrain_path: str = None, mode: str = None, model_name: str = None, frequent: int = 1,
                          display_log: bool = True):
    if pretrain_path is not None and os.path.isfile(os.path.join(pretrain_path, "run_config.json")):
        with open(os.path.join(pretrain_path, "run_config.json"), "r") as file:
            command_config = json.load(file)
        cut_type = command_config["cut_type"]
        problem_type = command_config["problem_type"]
        env_name = command_config["env_name"]
    else:
        assert "cut_type" is not None, "Need to provide the cut type"
        assert "problem_type" is not None, "Need to provide the problem type"
        assert "env_name" is not None, "Need to provide the env name"

    return explore_branch(problem_type, cut_type, env_name, pretrain_path=pretrain_path,
                          instance_path=instance_path, mode=mode, model_name=model_name,
                          frequent=frequent, display_log=display_log)


if __name__ == "__main__":
    app()
