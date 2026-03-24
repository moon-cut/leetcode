"""RNA secondary structure utilities.

Input: dot-bracket notation, e.g. "((..)).."
Output: graph edges suitable for message passing.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


Edge = Tuple[int, int]


class DotBracketParser:
    """Parse dot-bracket strings into undirected graph edges.

    Nodes are nucleotides in sequence order.
    Edges include:
    1) Backbone edges (i, i+1)
    2) Base-pair edges from matching parentheses.
    """

    _opening = "([{<"
    _closing = ")]} >".replace(" ", "")
    _match = {")": "(", "]": "[", "}": "{", ">": "<"}

    def parse(self, structure: str) -> List[Edge]:
        structure = structure.strip()
        if not structure:
            raise ValueError("Empty structure string.")

        n = len(structure)
        edges: List[Edge] = []
        stack = {ch: [] for ch in self._opening}

        for i in range(n - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))

        for idx, ch in enumerate(structure):
            if ch == ".":
                continue
            if ch in self._opening:
                stack[ch].append(idx)
                continue
            if ch in self._closing:
                op = self._match[ch]
                if not stack[op]:
                    raise ValueError(f"Unmatched closing bracket '{ch}' at position {idx}.")
                left = stack[op].pop()
                edges.append((left, idx))
                edges.append((idx, left))
                continue
            raise ValueError(f"Unsupported symbol '{ch}' in structure.")

        for opener, values in stack.items():
            if values:
                raise ValueError(f"Unmatched opening bracket '{opener}' at positions {values}.")

        return edges


def edge_index_from_edges(edges: Sequence[Edge]):
    """Return PyTorch Geometric style edge_index list-of-lists."""
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return [src, dst]
