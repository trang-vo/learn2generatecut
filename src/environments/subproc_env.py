from multiprocessing import Pipe
from multiprocessing.context import Process
from typing import Any, Callable, List, Optional, Union

import gymnasium as gym
from tianshou.env import BaseVectorEnv
from tianshou.env.utils import CloudpickleWrapper, ENV_TYPE
from tianshou.env.worker import SubprocEnvWorker
from tianshou.env.worker.subproc import ShArray, _setup_buf, _worker

from environments.env_name import ENV_NAME


def make_env(env_name, problem_type, cut_type, mode, config, **kwargs) -> Callable:
    def _init() -> gym.Env:
        env = ENV_NAME[env_name](problem_type=problem_type, cut_type=cut_type, mode=mode, **config, **kwargs)
        return env

    return _init


class MultiSubprocEnvWorker(SubprocEnvWorker):
    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=False)
        self.process.start()
        self.child_remote.close()
        super(SubprocEnvWorker, self).__init__(env_fn)


class MultiSubprocVectorEnv(BaseVectorEnv):
    def __init__(self, env_fns: List[Callable[[], ENV_TYPE]], **kwargs: Any) -> None:

        def worker_fn(fn: Callable[[], gym.Env]) -> MultiSubprocEnvWorker:
            return MultiSubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)
