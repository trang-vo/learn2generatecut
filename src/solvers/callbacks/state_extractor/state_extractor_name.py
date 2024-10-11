from .subtour import *
from .cycle import *

STATE_EXTRACTOR_NAME = {
    "subtour": {
        "default": SubtourStateExtractor,
        "SubtourStateExtractor": SubtourStateExtractor,
        "GraphormerSubtourStateExtractor": GraphormerSubtourStateExtractor,
    },
    "cycle": {
        "default": CycleStateExtractor,
        "CycleStateExtractor": CycleStateExtractor,
        "CycleStateExtractor2": CycleStateExtractor2,
        "GraphormerCycleStateExtractor": GraphormerCycleStateExtractor,
        "EmbedGraphormerCycleStateExtractor": EmbedGraphormerCycleStateExtractor,
    },
}