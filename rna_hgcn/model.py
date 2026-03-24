from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .hyperbolic_layers import HyperbolicGraphConv, PoincareBall


BASE_VOCAB = ["A", "U", "C", "G"]


class RNAInverseFoldingHGCN(nn.Module):
    """Map RNA secondary structure graph -> sequence logits.

    Input node features are learned positional embeddings.
    """

    def __init__(
        self,
        max_len: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 3,
        c: float = 1.0,
    ):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.manifold = PoincareBall(c=c)

        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(HyperbolicGraphConv(hidden_dim, hidden_dim, c=c))
        self.layers = nn.ModuleList(layers)

        self.readout = nn.Linear(hidden_dim, len(BASE_VOCAB))

    def forward(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_euc = self.pos_embed(positions)
        x_h = self.manifold.expmap0(x_euc)

        for layer in self.layers:
            x_h = layer(x_h, edge_index)

        x_euc_out = self.manifold.logmap0(x_h)
        logits = self.readout(x_euc_out)
        return logits

    @torch.no_grad()
    def decode(self, logits: torch.Tensor) -> str:
        ids = logits.argmax(dim=-1).tolist()
        return "".join(BASE_VOCAB[i] for i in ids)
