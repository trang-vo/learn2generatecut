from .base import *
from .env_callback import *


CALLBACK_NAME = {
    "BaseUserCallback": BaseUserCallback,
    "RLUserCallback": RLUserCallback,
    "EnvUserCallback": EnvUserCallback,
    "RandomUserCallback": RandomUserCallback,
    "PreprocessUserCallback": PreprocessUserCallback,
    "RecordRootCutCallback": RecordRootCutCallback,
    "MLUserCallback": MLUserCallback,
    "MiningUserCallback": MiningUserCallback,
    "SkipFactorUserCallback": SkipFactorUserCallback,
    "RainbowUserCallback": RainbowUserCallback,
    "HeuristicRainbowUserCallback": HeuristicRainbowUserCallback,
    "CombineHeuristicRainbowUserCallback": CombineHeuristicRainbowUserCallback
}
