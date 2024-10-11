import json
import os
from abc import ABC
from typing import *

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.type_aliases import GymEnv
import torch
from stable_baselines3.common.utils import get_linear_fn


class DumpLogsEveryNTimeSteps(BaseCallback):
    def __init__(self, n_steps=500, verbose=1):
        super(DumpLogsEveryNTimeSteps, self).__init__(verbose)
        self.check_freq = n_steps

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model._dump_logs()
        return True


class SaveReplayBufferEveryNTimeSteps(BaseCallback):
    def __init__(self, save_dir: str, n_steps=10000, verbose=1):
        super(SaveReplayBufferEveryNTimeSteps, self).__init__(verbose)
        self.check_freq = n_steps
        self.save_dir = save_dir

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if self.n_calls % self.check_freq == 0:
            save_path = os.path.join(self.save_dir, "buffer_{}_step.pkl".format(self.n_calls))
            self.model.save_replay_buffer(save_path)
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_time_mean",
                               safe_mean([ep_info["total_time"] for ep_info in self.model.ep_info_buffer]))
        return True


class EvalCheckpointCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    model_path = os.path.join(self.best_model_save_path, "best_model")
                    torch.save({"features_extractor": self.model.q_net.features_extractor.state_dict(),
                                "agent": self.model.q_net.q_net}, model_path)
                self.best_mean_reward = mean_reward

            if mean_reward > -1000:
                if self.best_model_save_path is not None:
                    model_path = os.path.join(self.best_model_save_path, "model_{}_steps.pt".format(self.n_calls))
                    torch.save({"features_extractor": self.model.q_net.features_extractor.state_dict(),
                                "agent": self.model.q_net.q_net}, model_path)

                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


def evaluate(env, model_path):
    model = DQN.load(model_path)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)


class SB3DQNAgent(ABC):
    def __init__(
            self,
            env: Union[GymEnv, str],
            env_kwargs: Dict[str, Any],
            model_class: str,
            model_kwargs: Dict[str, Any],
            model_folder: str,
            policy: str,
            learn_kwargs: Dict[str, Any],
            extractor_kwargs: Dict[str, Any] = None,
            pretrain_model_path: str = None,
            log_path: str = "../logs/",
    ):
        self.env = env
        self.env_kwargs = env_kwargs

        self.model_kwargs = model_kwargs
        if "learning_rate" in model_kwargs:
            if model_kwargs["learning_rate"] < 1e-9:
                model_kwargs["learning_rate"] = get_linear_fn(1e-3, 1e-5, 1)
                print("Dynamic learning rate")
        self.model_class = model_class
        self.model = MODEL_NAME[model_class](POLICY_NAME[policy], env, **model_kwargs)
        if not isinstance(self.model_kwargs["learning_rate"], float):
            self.model_kwargs["learning_rate"] = 0
        self.model_folder = model_folder
        self.learn_kwargs = learn_kwargs
        self.extractor_kwargs = extractor_kwargs
        self.log_path = log_path

        if not pretrain_model_path:
            print("CREATE NEW MODEL")
        else:
            print("LOAD PRETRAIN MODEL")
            model_dict = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
            self.model.q_net.q_net.load_state_dict(model_dict["agent"].state_dict())

        self._save_config()

    def _set_up_callback(self, eval_env: GymEnv):
        log_callback = DumpLogsEveryNTimeSteps(n_steps=1000)
        eval_callback = EvalCheckpointCallback(
            eval_env,
            best_model_save_path=self.log_path,
            log_path=self.log_path,
            eval_freq=self.learn_kwargs["eval_freq"],
            deterministic=True,
            render=False,
            n_eval_episodes=self.learn_kwargs["n_eval_episodes"],
        )
        buffer_callback = SaveReplayBufferEveryNTimeSteps(save_dir=self.log_path, n_steps=10)

        return [eval_callback, log_callback]

    def _save_config(self):
        with open(self.log_path + "config.json", "w") as file:
            json.dump(
                {
                    "model": self.model_class,
                    "dqn": self.model_kwargs,
                    "learn": self.learn_kwargs,
                    "env": self.env_kwargs,
                    "extractor": self.extractor_kwargs,
                },
                file,
            )

    def train(self, eval_env: GymEnv):
        logger = configure(self.log_path, ["stdout", "csv", "tensorboard"])
        callbacks = self._set_up_callback(eval_env=eval_env)

        self.model.set_logger(logger)
        self.model.learn(
            total_timesteps=self.learn_kwargs["total_timesteps"],
            log_interval=self.learn_kwargs["log_interval"],
            callback=callbacks,
        )
        self.model.save(self.log_path + self.model_folder + ".pt")
