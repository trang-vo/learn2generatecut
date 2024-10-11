from typing import *
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TsFeatureExtractor(nn.Module):
    def __init__(self, action_shape: int, solution_encoder: nn.Module = None, problem_encoder: nn.Module = None,
                 statistic_encoder: nn.Module = None, use_atoms: bool = True, num_atoms: int = 51, **kwargs) -> None:
        super().__init__()
        self.device = kwargs["device"]
        self.output_size = 0

        self.solution_encoder = solution_encoder
        self.problem_encoder = problem_encoder
        self.statistic_encoder = statistic_encoder

        if solution_encoder is not None:
            self.solution_encoder = solution_encoder.to(self.device)
            self.output_size += solution_encoder.output_size

        if problem_encoder is not None:
            self.problem_encoder = problem_encoder.to(self.device)
            self.output_size += problem_encoder.output_size

        if statistic_encoder is not None:
            self.statistic_encoder = statistic_encoder.to(self.device)
            self.output_size += statistic_encoder.output_size

        self.use_atoms = use_atoms
        self.num_atoms = num_atoms
        action_dim = int(np.prod(action_shape)) * num_atoms
        self.linear = nn.Linear(self.output_size, action_dim).to(self.device)
        print("Feature extractor in device", self.device)

    def forward(self, observations, state=None, info={}):
        features = []
        # print("Observations", observations)
        if self.solution_encoder is not None:
            sup_vec = self.solution_encoder(observations["solution"])
            features.append(sup_vec)

        if self.problem_encoder is not None:
            ori_vec = self.problem_encoder(observations["problem"])
            features.append(ori_vec)

        if self.statistic_encoder is not None:
            statistic_vec = self.statistic_encoder(observations["statistic"])
            features.append(statistic_vec)

        x = torch.cat(features, dim=1)
        if self.use_atoms:
            x = self.linear(x)
            bsz = x.shape[0]

            if self.num_atoms > 1:
                x = x.view(bsz, -1, self.num_atoms)
                x = F.softmax(x, dim=-1)

        return x, state
