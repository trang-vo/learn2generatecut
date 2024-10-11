import _pickle
import os
import socket
import time
from typing import *
import queue
import warnings

import gymnasium as gym
from gymnasium import spaces
from torch.multiprocessing import Queue, Process
import numpy as np

from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.state_extractor.state_extractor_name import STATE_EXTRACTOR_NAME
from solvers.base import CALLBACK_NAME
from solvers.callbacks.env_callback import RecordBranchCallback, DFSRandomNodeCallback
from problems.problem_name import PROBLEM_NAME
from config import EnvConfig

from solvers.callbacks.env_callback import RandomStartBranchCallback


class BaseCutEnv(gym.Env):
    def __init__(
            self,
            problem_type: str,
            cut_type: str,
            data_folder: str,
            space_config: Dict[str, Any],
            episode_config: Dict[str, Any],
            mdp_type: str,
            state_extractor_class: str,
            state_components: List[str] = ("solution", "problem", "statistic"),
            user_callback_type: str = "EnvUserCallback",
            mode: str = "train",
            result_path: str = "",
            k_nearest_neighbors: int = 10,
            *args, **kwargs
    ) -> None:
        self.problem_type = problem_type
        self.cut_type = cut_type
        self.space_config = space_config
        self.episode_config = episode_config
        self.mdp_type = mdp_type
        self.state_extractor_class = state_extractor_class
        self.state_components = state_components
        self.mode = mode
        self.result_path = result_path
        self.k_nearest_neighbors = k_nearest_neighbors

        self.problem = None
        self.solver = None
        self.solver_proc = None
        self.action_queue = None
        self.state_queue = None
        self.done = False
        self.total_time = 0
        self.user_callback = None
        self.last_state = None
        self.prior_buffer = False

        self.init_config = EnvConfig(space_config)
        self.action_space = spaces.Discrete(2)

        self.data_folder = data_folder
        if mode == "train":
            self.train_folder = os.path.join(self.data_folder, "train")
            self.train_instances = [file for file in os.listdir(self.train_folder) if
                                    os.path.isfile(os.path.join(self.train_folder, file))]
        self.eval_folder = os.path.join(self.data_folder, "eval")
        self.eval_instances = [file for file in os.listdir(self.eval_folder) if
                               os.path.isfile(os.path.join(self.eval_folder, file))]

        self.num_steps = 0
        self.user_callback_type = user_callback_type
        self.user_callback_class = CALLBACK_NAME[user_callback_type]

        self.initial_probability = kwargs.get("initial_probability", 1)

    def cplex_solve(self):
        self.solver.solve()
        if self.mode == "eval":
            with open(self.result_path, "a") as file:
                file.write(
                    "{},{:.2f},{:.4f},{:.4f},{},{}\n".format(self.problem.graph.graph["name"],
                                                             self.user_callback.curr_state.gap,
                                                             self.user_callback.total_reward,
                                                             self.user_callback.total_cuts,
                                                             self.user_callback.actions[0],
                                                             self.user_callback.actions[1]))

    def get_instance_path(self):
        instance_path = None
        if self.mode == "train":
            i = np.random.randint(0, len(self.train_instances))
            instance_path = os.path.join(
                self.train_folder, self.train_instances[i]
            )
        elif self.mode == "eval":
            i = np.random.randint(0, len(self.eval_instances))
            instance_path = os.path.join(
                self.eval_folder, self.eval_instances[i]
            )

        return instance_path

    def get_log_path_evaluation(self):
        tmp = self.result_path.split("/")
        logdir = os.path.join(*tmp[:-1])
        try:
            if not os.path.isdir(os.path.join(logdir, "eval_log")):
                os.makedirs(os.path.join(logdir, "eval_log"))
                print("Created log path", os.path.join(logdir, "eval_log"))
        except FileExistsError:
            pass

        log_path = os.path.join(
            logdir,
            "eval_log",
            "{}_{}.log".format(self.problem.graph.graph["name"], self.num_steps),
        )

        return log_path

    def create_mip_solver(self, **kwargs):
        log_path = ""
        if self.mode == "eval":
            assert self.result_path != ""
            log_path = self.get_log_path_evaluation()

        self.solver = SOLVER_NAME[self.problem_type](
            problem=self.problem,
            cut_type=self.cut_type,
            display_log=kwargs["display_log"] if "display_log" in kwargs else False,
            log_path=log_path,
            time_limit=3600 if self.mode == "train" else 1800,
        )
        self.solver.parameters.mip.strategy.nodeselect.set(0)
        state_extractor = STATE_EXTRACTOR_NAME[self.cut_type][self.state_extractor_class](self.state_components,
                                                                                          self.mdp_type,
                                                                                          padding=True)
        state_extractor.initialize_original_graph(self.problem, self.solver.edge2idx, k=self.k_nearest_neighbors)

        host_name = socket.gethostname()
        root_cut_path = os.path.join(self.data_folder, self.mode, "root_cuts", host_name,
                                     self.problem.graph.name + ".pkl")
        if os.path.isfile(root_cut_path):
            with open(root_cut_path, "rb") as file:
                root_cuts = iter(_pickle.load(file))
        else:
            warnings.warn(f"Cannot find the root cut file for instance {root_cut_path}")
            root_cuts = None
        self.user_callback = self.solver.register_callback(self.user_callback_class)
        user_cb_kwargs = {
            "state_extractor": state_extractor,
            "state_queue": self.state_queue,
            "action_queue": self.action_queue,
            "env_mode": self.mode,
            "mdp_type": self.mdp_type,
            "config": self.episode_config,
            "logger": self.solver.logger,
            "root_cuts": root_cuts,
            "initial_probability": self.initial_probability
        }
        self.user_callback.set_attribute(self.solver.separator, **user_cb_kwargs)

        self.solver.register_callback(RandomStartBranchCallback)
        if "random_path" in kwargs and kwargs["random_path"]:
            self.solver.register_callback(RecordBranchCallback)
            self.solver.register_callback(DFSRandomNodeCallback)

    def set_mip_solver(self, instance_path=None, **kwargs):
        if self.solver_proc is not None:
            self.solver_proc.terminate()
            self.action_queue.close()
            self.state_queue.close()

        self.action_queue = Queue()
        self.state_queue = Queue()
        self.done = False

        if instance_path is None:
            instance_path = self.get_instance_path()
        self.problem = PROBLEM_NAME[self.problem_type](instance_path)

        print("Processing instance", instance_path)
        self.create_mip_solver(**kwargs)

        self.solver_proc = Process(target=self.cplex_solve, args=())
        self.solver_proc.daemon = True
        self.solver_proc.start()

    def reset(self, instance_path=None, **kwargs):
        self.set_mip_solver(instance_path, **kwargs)

        while True:
            try:
                obs, _, done, _ = self.state_queue.get(timeout=5)
                return obs, {}
            except queue.Empty:
                print("Waiting an initial state")
                if not self.solver_proc.is_alive():
                    self.set_mip_solver(**kwargs)

    def step(self, action: int):
        self.action_queue.put(action)

        while self.solver_proc.is_alive():
            try:
                obs, reward, done, info = self.state_queue.get(timeout=5)
                self.last_state = obs
                self.total_time = info.get("total_time", 0)
                # Wait to write results to file if mode is eval
                if self.mode == "eval" and done:
                    time.sleep(1)
                self.num_steps += 1
                return obs, reward, done, False, info
            except queue.Empty:
                print("Queue is empty")

        done = True
        reward = 0
        info = {"terminal_observation": self.last_state, "total_time": self.total_time}
        if self.mode == "eval":
            time.sleep(1)
        self.num_steps += 1
        return self.last_state, reward, done, False, info

    def seed(self, seed):
        np.random.seed(seed)
