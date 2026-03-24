from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .model import BASE_VOCAB
from .structure import DotBracketParser


BASE2ID = {b: i for i, b in enumerate(BASE_VOCAB)}


@dataclass
class RNASample:
    structure: str
    sequence: str


class RNAGraphBuilder:
    def __init__(self):
        self.parser = DotBracketParser()

    def build(self, sample: RNASample):
        if len(sample.structure) != len(sample.sequence):
            raise ValueError("Structure and sequence lengths must match.")

        n = len(sample.structure)
        positions = torch.arange(n, dtype=torch.long)
        edges = self.parser.parse(sample.structure)
        edge_index = torch.tensor(edges, dtype=torch.long).T
        labels = torch.tensor([BASE2ID[b] for b in sample.sequence], dtype=torch.long)
        return positions, edge_index, labels


def tiny_demo_dataset() -> List[RNASample]:
    return [
        RNASample(structure="(((...)))", sequence="GCGAAACGC"),
        RNASample(structure="((..((..))))", sequence="AUGCGCAAUGCU"),
        RNASample(structure="....", sequence="AUCG"),
    ]
