from typing import *

import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]


class StandardNet(nn.Module):
    def __init__(
            self,
            feature_extractor: ModuleType,
            action_shape: Union[int, Sequence[int]] = 0,
            hidden_sizes: Sequence[int] = (),
            device: Union[str, int, torch.device] = "cpu",
            *args, **kwargs,
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor
        self.policy_net = nn.Sequential(
            nn.Linear(in_features=feature_extractor.output_size, out_features=hidden_sizes[0], bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1], bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_sizes[1], out_features=action_shape, bias=True)
        )
        self.policy_net.to(device)

    def forward(self, obs, state=None, info={}):
        features, state = self.feature_extractor(obs, state, info)
        logits = self.policy_net(features)

        return logits, state
