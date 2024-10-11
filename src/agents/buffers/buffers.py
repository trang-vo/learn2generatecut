from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np
from copy import deepcopy

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class PriorDictReplayBuffer(DictReplayBuffer):
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        samples = deepcopy(self.observations["prior"].ravel()[:upper_bound])
        priority_ratio = int(samples.size / np.sum(samples)) if np.sum(samples) > 0 else 1
        print("PRIORITY RATIO", priority_ratio, samples.size, np.sum(samples))
        samples *= priority_ratio
        samples = np.where(samples > 0, samples, 1)
        probabilities = samples / samples.sum()
        batch_inds = np.random.choice(range(len(samples)), batch_size, p=probabilities)
        return self._get_samples(batch_inds, env=env)

