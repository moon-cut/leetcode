"""Train a minimal HGCN for RNA inverse folding demo.

Usage:
    python train_rna_hgcn.py --epochs 200
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from rna_hgcn.data import RNAGraphBuilder, tiny_demo_dataset
from rna_hgcn.model import RNAInverseFoldingHGCN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = tiny_demo_dataset()
    builder = RNAGraphBuilder()

    max_len = max(len(x.structure) for x in data) + 8
    model = RNAInverseFoldingHGCN(
        max_len=max_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        c=1.0,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for sample in data:
            pos, edge_index, labels = builder.build(sample)
            pos, edge_index, labels = pos.to(device), edge_index.to(device), labels.to(device)

            logits = model(pos, edge_index)
            loss = F.cross_entropy(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss={total_loss / len(data):.4f}")

    model.eval()
    with torch.no_grad():
        for sample in data:
            pos, edge_index, _ = builder.build(sample)
            pos, edge_index = pos.to(device), edge_index.to(device)
            pred = model.decode(model(pos, edge_index).cpu())
            print(f"structure={sample.structure} | pred={pred} | target={sample.sequence}")


if __name__ == "__main__":
    main()
