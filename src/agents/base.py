from abc import ABC
from typing import *

# from ..environments.base import BaseCutEnv


class Agent(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def _save_train_config(self, log_path: str, *args, **kwargs):
        raise NotImplementedError

    def train(self, train_env, eval_env, learn_config, log_path, model_folder, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, pretrain_path: str = None, *args, **kwargs):
        raise NotImplementedError

