import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        sup_feature_extractor: nn.Module,
        ori_feature_extractor: nn.Module,
        statistic_extractor: nn.Module,
        features_dim: int = 1,
        **kwargs,
    ):
        super().__init__(observation_space, features_dim)
        self.device = kwargs["device"]
        self.sup_feature_extractor = sup_feature_extractor.to(self.device)
        self.ori_feature_extractor = ori_feature_extractor.to(self.device)
        self.statistic_extractor = statistic_extractor.to(self.device)

        self._features_dim = (
            self.sup_feature_extractor.output_size
            + self.ori_feature_extractor.output_size
            + self.statistic_extractor.output_size
        )

    def forward(self, observations):
        sup_vec = self.sup_feature_extractor(
            observations["sup_node_feature"],
            observations["sup_edge_index"],
            observations["sup_edge_feature"],
            observations["sup_lens"],
        )

        ori_vec = self.ori_feature_extractor(
            observations["ori_node_feature"],
            observations["ori_edge_index"],
            observations["ori_edge_feature"],
            observations["ori_lens"],
        )

        statistic_vec = self.statistic_extractor(observations["statistic"])

        features = [sup_vec, ori_vec, statistic_vec]
        x = torch.cat(features, dim=1)

        return x


class EvalFeatureExtractor(nn.Module):
    def __init__(
        self,
        sup_feature_extractor: nn.Module,
        ori_feature_extractor: nn.Module,
        statistic_extractor: nn.Module,
        **kwargs
    ) -> None:
        super().__init__()
        self.device = kwargs["device"]
        self.sup_feature_extractor = sup_feature_extractor.to(self.device)
        self.ori_feature_extractor = ori_feature_extractor.to(self.device)
        self.statistic_extractor = statistic_extractor.to(self.device)
        print("Eval feature extractor in device", self.device)

    def forward(self, observations):
        sup_vec = self.sup_feature_extractor(
            observations["sup_node_feature"],
            observations["sup_edge_index"],
            observations["sup_edge_feature"],
            observations["sup_lens"],
        )

        ori_vec = self.ori_feature_extractor(
            observations["ori_node_feature"],
            observations["ori_edge_index"],
            observations["ori_edge_feature"],
            observations["ori_lens"],
        )

        statistic_vec = self.statistic_extractor(observations["statistic"])

        features = [sup_vec, ori_vec, statistic_vec]
        x = torch.cat(features, dim=1)

        return x


class TsFeatureExtractor(nn.Module):
    def __init__(
        self,
        action_shape: int,
        sup_feature_extractor: nn.Module = None,
        ori_feature_extractor: nn.Module = None,
        statistic_extractor: nn.Module = None,
        num_atoms: int = 1,
        **kwargs
    ) -> None:
        super().__init__()
        self.device = kwargs["device"]
        self.output_size = 0

        self.sup_feature_extractor = sup_feature_extractor
        self.ori_feature_extractor = ori_feature_extractor
        self.statistic_extractor = statistic_extractor

        if sup_feature_extractor is not None:
            self.sup_feature_extractor = sup_feature_extractor.to(self.device)
            self.output_size += sup_feature_extractor.output_size

        if ori_feature_extractor is not None:
            self.ori_feature_extractor = ori_feature_extractor.to(self.device)
            self.output_size += ori_feature_extractor.output_size

        if statistic_extractor is not None:
            self.statistic_extractor = statistic_extractor.to(self.device)
            self.output_size += statistic_extractor.output_size

        self.num_atoms = num_atoms
        action_dim = int(np.prod(action_shape)) * num_atoms
        self.linear = nn.Linear(self.output_size, action_dim).to(self.device)
        print("Eval feature extractor in device", self.device)

    def forward(self, observations, state=None, info={}):
        features = []
        if self.sup_feature_extractor is not None:
            sup_vec = self.sup_feature_extractor(
                observations["sup_node_feature"],
                observations["sup_edge_index"],
                observations["sup_edge_feature"],
                observations["sup_lens"],
            )
            features.append(sup_vec)

        if self.ori_feature_extractor is not None:
            ori_vec = self.ori_feature_extractor(
                observations["ori_node_feature"],
                observations["ori_edge_index"],
                observations["ori_edge_feature"],
                observations["ori_lens"],
            )
            features.append(ori_vec)

        if self.statistic_extractor is not None:
            statistic_vec = self.statistic_extractor(observations["statistic"])
            features.append(statistic_vec)

        x = torch.cat(features, dim=1)
        x = self.linear(x)
        bsz = x.shape[0]

        if self.num_atoms > 1:
            x = x.view(bsz, -1, self.num_atoms)
            x = F.softmax(x, dim=-1)

        return x, state