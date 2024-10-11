from .noisy_net import NoisyNet
from .standard_net import StandardNet

POLICY_NET_NAME = {
    "NoisyNet": NoisyNet,
    "StandardNet": StandardNet,
}