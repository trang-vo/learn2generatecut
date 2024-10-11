from typing import *

import numpy as np
import torch as th
from gym.vector.utils import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update, get_linear_fn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn
from stable_baselines3.common.type_aliases import Schedule, GymEnv


def softmax_temperature(x: th.Tensor, tau: float = 1e-4):
    x_rel: th.Tensor = x / tau - x.max() / tau
    e_x: th.Tensor = th.exp(x_rel)
    return e_x / e_x.sum()


def softmax_selection(prob: th.Tensor):
    p = th.cumsum(prob, dim=0)
    sample = np.random.rand()
    action = th.where(p > sample)[0][0]
    return action


class SoftmaxQNetwork(QNetwork):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )
        self.temperature = 1e-4

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if not deterministic:
            q_values = self(observation)
            prob: th.Tensor = th.stack([softmax_temperature(x_i, self.temperature) for x_i in th.unbind(q_values, dim=0)],
                                       dim=0)
            # softmax selection action
            actions: th.Tensor = th.stack([softmax_selection(x_i) for x_i in th.unbind(prob, dim=0)], dim=0)
        else:
            q_values = self(observation)
            # Greedy action
            actions = q_values.argmax(dim=1).reshape(-1)

        return actions


class SoftmaxDQNPolicy(DQNPolicy):
    def make_q_net(self) -> SoftmaxQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return SoftmaxQNetwork(**net_args).to(self.device)


class SoftmaxDQN(DQN):
    def __init__(
            self,
            policy: Union[str, Type[DQNPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 50000,
            batch_size: int = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 4,
            gradient_steps: int = 1,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        print("device is", device)
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.temperature_schedule = get_linear_fn(
            exploration_initial_eps,
            exploration_final_eps,
            exploration_fraction,
        )

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

        self.policy.q_net.temperature = self.temperature_schedule(self._current_progress_remaining)
        self.logger.record("rollout/softmax_temperature", self.policy.q_net.temperature)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
