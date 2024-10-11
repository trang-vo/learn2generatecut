from typing import *

import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]


class NoisyNet(nn.Module):
    def __init__(
            self,
            feature_extractor: ModuleType,
            action_shape: Union[int, Sequence[int]] = 0,
            hidden_sizes: Sequence[int] = (),
            norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
            norm_args: Optional[ArgsType] = None,
            activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
            act_args: Optional[ArgsType] = None,
            device: Union[str, int, torch.device] = "cpu",
            softmax: bool = False,
            concat: bool = False,
            num_atoms: int = 51,
            noisy_std: float = 0.1,
            *args, **kwargs,
    ) -> None:
        super().__init__()

        def noisy_linear(x, y):
            return NoisyLinear(x, y, noisy_std)

        self.feature_extractor = feature_extractor
        self.policy_net = Net(
            state_shape=feature_extractor.output_size,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            device=device,
            softmax=True,
            num_atoms=num_atoms,
            norm_layer=norm_layer,
            norm_args=norm_args,
            activation=activation,
            act_args=act_args,
            concat=concat,
            dueling_param=({
                               "linear_layer": noisy_linear
                           }, {
                               "linear_layer": noisy_linear
                           }),
        ).to(device)

    def forward(self, obs, state=None, info={}):
        features, state = self.feature_extractor(obs, state, info)
        logits, state = self.policy_net(features, state, info)

        return logits, state
