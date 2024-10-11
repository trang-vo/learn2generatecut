from time import time

import numpy as np
import torch
import torch_geometric
import fairseq
from fairseq import utils

from .graphormer_src.modules.graphormer_graph_encoder import GraphormerGraphEncoder
from .graphormer_src.preprocess.collator import collator


class GraphormerExtractor(GraphormerGraphEncoder):
    def __init__(
            self,
            num_node_features: int,
            num_edge_features: int,
            num_atoms: int,
            num_in_degree: int,
            num_out_degree: int,
            num_edges: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,
            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            pre_layernorm: bool = False,
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            device: str = "cpu",
    ) -> None:
        super().__init__(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            layerdrop=layerdrop,
            encoder_normalize_before=encoder_normalize_before,
            pre_layernorm=pre_layernorm,
            apply_graphormer_init=apply_graphormer_init,
            activation_fn=activation_fn,
            embed_scale=embed_scale,
            freeze_embeddings=freeze_embeddings,
            n_trans_layers_to_freeze=n_trans_layers_to_freeze,
            export=export,
            traceable=traceable,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.multi_hop_max_dist = multi_hop_max_dist
        self.output_size = embedding_dim
        self.lm_head_transform_weight = torch.nn.Linear(embedding_dim, embedding_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = fairseq.modules.LayerNorm(embedding_dim)
        self.device = device

    def forward(self, items: np.ndarray):
        items = items.tolist()
        for idx in range(len(items)):
            items[idx].idx = idx

        batched_data = collator(items, multi_hop_max_dist=self.multi_hop_max_dist)
        for key, tensor in batched_data.items():
            if torch.is_tensor(tensor):
                batched_data[key] = tensor.to(torch.device(self.device))

        _, graph_rep = super().forward(batched_data)
        graph_rep = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(graph_rep)))
        graph_rep = torch.nn.functional.normalize(graph_rep)

        return graph_rep
