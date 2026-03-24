"""Minimal hyperbolic layers inspired by HGCN ideas.

The implementation keeps operations simple and educational.
For production use, consider replacing these ops with geoopt/torch_hyperbolic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def artanh(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class PoincareBall:
    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        self.c = c
        self.eps = eps

    @property
    def sqrt_c(self):
        return self.c ** 0.5

    def project(self, x: torch.Tensor) -> torch.Tensor:
        max_norm = (1.0 - self.eps) / self.sqrt_c
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.tanh(self.sqrt_c * v_norm) / (self.sqrt_c * v_norm)
        return self.project(scale * v)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = artanh(self.sqrt_c * x_norm) / (self.sqrt_c * x_norm)
        return scale * x

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + (c**2) * x2 * y2
        return self.project(num / den.clamp_min(self.eps))

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_tan = self.logmap0(x)
        mx_tan = x_tan @ m.T
        return self.expmap0(mx_tan)


class HyperbolicGraphConv(nn.Module):
    """A simple hyperbolic message-passing layer.

    Steps:
    1) map nodes to tangent space at origin
    2) aggregate neighbors in Euclidean tangent space
    3) linear transform
    4) map back to Poincare ball
    """

    def __init__(self, in_dim: int, out_dim: int, c: float = 1.0):
        super().__init__()
        self.manifold = PoincareBall(c=c)
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x_h: [N, D] in hyperbolic space
        # edge_index: [2, E]
        n = x_h.size(0)
        src, dst = edge_index

        x_tan = self.manifold.logmap0(x_h)
        agg = torch.zeros_like(x_tan)
        agg.index_add_(0, dst, x_tan[src])

        deg = torch.bincount(dst, minlength=n).float().unsqueeze(-1).clamp_min(1.0)
        agg = agg / deg

        out_tan = self.lin(agg) + self.bias
        out_h = self.manifold.expmap0(out_tan)
        return out_h
