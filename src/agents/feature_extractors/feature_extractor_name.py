from .component_encoders.gine import GINEGraphExtractor
from .component_encoders.gnn import GNNGraphExtractor
from .component_encoders.mlp import MLP, MLPOneLayer
from .component_encoders.graphormer import *
from .component_encoders.embed_mlp import EmbedMLP


FEATURE_EXTRACTOR_NAME = {
    "GNNGraphExtractor": GNNGraphExtractor,
    "GINEGraphExtractor": GINEGraphExtractor,
    "MLP": MLP,
    "MLPOneLayer": MLPOneLayer,
    "GraphormerExtractor": GraphormerExtractor,
    "EmbedMLP": EmbedMLP,
}
