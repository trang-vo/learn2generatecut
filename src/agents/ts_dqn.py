import os
import time
from typing import *

import numpy as np
import torch.nn as nn
import torch.optim
from tianshou.utils.net.common import Net
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from agents.base import Agent
from environments.base import BaseCutEnv


class RainbowPolicyGPU(ts.policy.RainbowPolicy):
    def assign_device(self, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        if device is not "cpu":
            self.support = torch.nn.Parameter(
                torch.linspace(self._v_min, self._v_max, self._num_atoms, device=device),
                requires_grad=False,
            )


class DQNPolicyGPU(ts.policy.DQNPolicy):
    def assign_device(self, device: str = "cuda:0"):
        return None


DQN_POLICY = {
    "RainbowPolicyGPU": RainbowPolicyGPU,
    "DQNPolicyGPU": DQNPolicyGPU,
}


class TsDQN(Agent):
    def __init__(
            self,
            net: nn.Module = None,
            optimizer_config: Dict[str, Any] = {},
            policy_config: Dict[str, Any] = {},
            device: str = "cuda",
            *args, **kwargs,
    ):
        if net is not None:
            self.net = net
        elif "state_shape" in kwargs and "action_shape" in kwargs:
            state_shape = kwargs["state_shape"]
            action_shape = kwargs["action_shape"]
            self.net = Net(state_shape, action_shape, num_atoms=51, softmax=True)
        else:
            raise KeyError("Need to provide either pre-defined network or the state and action shapes")

        lr = optimizer_config.get("lr", 1e-4)
        weight_decay = optimizer_config.get("weight_decay", 0.01)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        if policy_config["name"] == "DQNPolicyGPU":
            self.policy = DQNPolicyGPU(
                model=self.net,
                optim=self.optimizer,
                discount_factor=policy_config.get("discount_factor", 1),
                estimation_step=policy_config.get("estimation_step", 3),
                target_update_freq=policy_config.get("target_update_freq", 500),
            )
        elif policy_config["name"] == "RainbowPolicyGPU":
            self.policy = RainbowPolicyGPU(
                model=self.net,
                optim=self.optimizer,
                num_atoms=policy_config.get("num_atoms", 51),
                discount_factor=policy_config.get("discount_factor", 1),
                estimation_step=policy_config.get("estimation_step", 3),
                target_update_freq=policy_config.get("target_update_freq", 200),
                v_min=policy_config.get("v_min", -20),
                v_max=policy_config.get("v_max", 0)
            )
        self.policy.assign_device(device=device if torch.cuda.is_available() else "cpu")

    @staticmethod
    def lin_exploration_rate_scheduler(start: float, end: float, max_epochs: int):
        def func(nb_epochs: int, step_idx: int):
            value = start - (start - end) / max_epochs * nb_epochs
            return value

        return func

    @staticmethod
    def exp_exploration_rate_scheduler(start: float, end: float, decay: int):
        def func(nb_epochs: int, step_idx: int):
            value = end + (start - end) * np.exp(-1. * step_idx / decay)
            return value

        return func

    def train(
            self,
            train_env: Union[BaseCutEnv, List[BaseCutEnv]],
            eval_env: Union[BaseCutEnv, List[BaseCutEnv]],
            log_path: str,
            learn_config: Dict[str, Any] = {},
            *args, **kwargs,
    ):
        if learn_config.get("pretrain_path", None) is not None:
            self.policy.load_state_dict(torch.load(learn_config["pretrain_path"]))
            print(f'Load pretrain agent from {learn_config["pretrain_path"]}')

        n_cpu = learn_config.get("n_cpu", 1)
        max_epoch = learn_config.get("max_epoch", 10)
        step_per_epoch = learn_config.get("step_per_epoch", 1000)
        reward_threshold = learn_config.get("reward_threshold", 0)

        # Exploration rate decay
        decay_strategy = learn_config["epsilon_greedy"]["decay_strategy"]
        epsilon_init = learn_config["epsilon_greedy"]["epsilon_init"]
        epsilon_final = learn_config["epsilon_greedy"]["epsilon_final"]
        epsilon_decay = learn_config["epsilon_greedy"]["epsilon_decay"]
        if decay_strategy == "linear":
            print("The exploration rate decay type is linear")
            epsilon_scheduler = self.lin_exploration_rate_scheduler(epsilon_init, epsilon_final, max_epoch)
        else:
            print("The exploration rate decay type is exponential")
            epsilon_scheduler = self.exp_exploration_rate_scheduler(epsilon_init, epsilon_final, epsilon_decay)

        # Define the prioritized beta annealing schedule
        beta_scheduler = None
        buffer_size = learn_config["buffer_replay"]["buffer_size"]
        if learn_config["buffer_replay"]["type"] == "prioritized":
            print("The buffer type is prioritized")
            alpha = learn_config["buffer_replay"]["alpha"]
            beta_init = learn_config["buffer_replay"]["beta"]
            beta_final = learn_config["buffer_replay"]["beta_final"]
            beta_annealing = learn_config["buffer_replay"]["beta_annealing"]
            if learn_config["buffer_replay"]["beta_annealing_strategy"] == "linear":
                print("The beta annealing type is linear")
                beta_scheduler = self.lin_exploration_rate_scheduler(beta_init, beta_final, beta_annealing)
            else:
                print("The beta annealing type is exponential")
                beta_scheduler = self.exp_exploration_rate_scheduler(beta_init, beta_final, beta_annealing)

            train_buffer = ts.data.PrioritizedVectorReplayBuffer(
                buffer_size,
                buffer_num=len(train_env),
                alpha=alpha,
                beta=beta_init,
                weight_norm=True
            )
        else:
            train_buffer = ts.data.VectorReplayBuffer(buffer_size, n_cpu)

        initial_buffer_path = learn_config.get("initial_buffer", None)
        if initial_buffer_path is not None and os.path.isfile(initial_buffer_path):
            train_buffer = train_buffer.load_hdf5(initial_buffer_path)
            print("Load the initial replay buffer from", initial_buffer_path)

        train_collector = ts.data.Collector(
            policy=self.policy,
            env=train_env,
            buffer=train_buffer,
            exploration_noise=True,
        )

        eval_collector = ts.data.Collector(
            policy=self.policy,
            env=eval_env,
            exploration_noise=False,
        )

        logger = TensorboardLogger(SummaryWriter(log_path))

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, "best_model.pth"))

        def stop_fn(mean_rewards):
            return mean_rewards >= reward_threshold

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            ckpt_path = os.path.join(log_path, "model_{}_epochs.pth".format(epoch))
            torch.save(self.policy.state_dict(), ckpt_path)

            optim_path = os.path.join(log_path, "optim_{}_epochs.pth".format(epoch))
            torch.save(self.policy.optim.state_dict(), optim_path)

        def train_fn(epoch, env_step):
            curr_eps = epsilon_scheduler(epoch, env_step)
            curr_eps = curr_eps if curr_eps > epsilon_final else epsilon_final
            self.policy.set_eps(curr_eps)

            if beta_scheduler is not None:
                curr_beta = beta_scheduler(epoch, env_step)
                curr_beta = curr_beta if curr_beta < beta_final else beta_final
                train_buffer.set_beta(curr_beta)

        results = ts.trainer.offpolicy_trainer(
            self.policy, train_collector, eval_collector, max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=learn_config.get("step_per_collect", 16),
            update_per_step=learn_config.get("update_per_step", 1),
            episode_per_test=learn_config.get("episode_per_test", 1),
            batch_size=learn_config.get("batch_size", 256),
            train_fn=lambda epoch, env_step: train_fn(epoch, env_step),
            stop_fn=lambda mean_rewards: stop_fn(mean_rewards),
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
        )

        print(f'Finished training! Use {results["duration"]}')

        return results

    def evaluate(self, eval_env, instance_path: str, pretrain_path: str = "", model_name: str = None,
                 display_log: bool = True, *args, **kwargs):
        if pretrain_path != "" and not os.path.isdir(pretrain_path):
            raise FileNotFoundError(f"The pretrain path {pretrain_path} is not valid directory\n")

        if model_name is None:
            model_name = "best_model.pth"

        model_path = os.path.join(pretrain_path, model_name)
        assert os.path.isfile(model_path), f"{model_path} is invalid dir"

        if pretrain_path != "":
            self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()

        obs, _ = eval_env.reset(display_log=display_log, instance_path=instance_path)
        data = ts.data.batch.Batch({"obs": ts.data.batch.Batch([obs]), "info": {}})

        s = time.time()
        terminated = False
        info = {}
        while not terminated:
            act = self.policy(data)["act"][0]
            obs, reward, terminated, _, info = eval_env.step(act)
            data["obs"] = ts.data.batch.Batch([obs])
            data["info"] = info
            data["reward"] = reward

        return time.time() - s, data["reward"], info.get("node_id", 0), info.get("total_cuts", 0)
