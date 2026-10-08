from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


V2_ELEMENT_TYPES = (
    "gene",
    "intergenic",
    "unclassified_window",
    "promoter",
    "attenuator",
    "origin",
    "insertion_sequence",
    "prophage_boundary",
)


@dataclass(frozen=True)
class GenomeWindow:
    genome_id: str
    start: int
    end: int
    sequence: str
    element_type: str
    evidence_level: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("genome window coordinates must satisfy 0 <= start < end")
        if self.end - self.start != len(self.sequence):
            raise ValueError("window coordinates must match the sequence length")
        if self.element_type not in V2_ELEMENT_TYPES:
            raise ValueError(f"unsupported v2 genomic element type: {self.element_type}")
        if self.evidence_level not in {"E1", "E2", "E3", "E4"}:
            raise ValueError("evidence_level must be E1, E2, E3, or E4")


def make_dna_windows(
    genome_id: str,
    sequence: str,
    window_size: int = 4096,
    overlap: int = 512,
) -> list[GenomeWindow]:
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("require window_size > overlap >= 0")
    dna = "".join(sequence.upper().split())
    invalid = set(dna) - set("ACGTN")
    if invalid:
        raise ValueError(f"DNA sequence contains invalid symbols: {sorted(invalid)}")
    stride = window_size - overlap
    windows = []
    for start in range(0, len(dna), stride):
        end = min(len(dna), start + window_size)
        windows.append(
            GenomeWindow(
                genome_id=genome_id,
                start=start,
                end=end,
                sequence=dna[start:end],
                element_type="unclassified_window",
                evidence_level="E3",
            )
        )
        if end == len(dna):
            break
    return windows


class Version2WindowProjector(nn.Module):
    """Frozen Evo2 embeddings are supplied by the optional upstream adapter."""

    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, window_embeddings: torch.Tensor) -> torch.Tensor:
        if window_embeddings.ndim != 2:
            raise ValueError("window_embeddings must have shape [windows, dimensions]")
        return self.projector(window_embeddings)


def validate_element_nodes(windows: Sequence[GenomeWindow]) -> None:
    seen: set[tuple[str, int, int]] = set()
    for window in windows:
        key = (window.genome_id, window.start, window.end)
        if key in seen:
            raise ValueError(f"duplicate genome window: {key}")
        seen.add(key)


def element_gene_edges(
    windows: Iterable[GenomeWindow],
    gene_intervals: Iterable[tuple[str, str, int, int]],
    maximum_gap: int = 60,
) -> list[tuple[str, str, float, str]]:
    """Return typed proximity edges; these are annotations, not functional proof."""
    genes = list(gene_intervals)
    annotatable_elements = {
        "promoter",
        "attenuator",
        "origin",
        "insertion_sequence",
        "prophage_boundary",
    }
    edges: list[tuple[str, str, float, str]] = []
    for window in windows:
        if window.element_type not in annotatable_elements:
            continue
        for genome_id, gene_id, start, end in genes:
            if genome_id != window.genome_id:
                continue
            gap = max(0, max(start - window.end, window.start - end))
            if gap <= maximum_gap:
                edges.append((gene_id, f"{window.genome_id}:{window.start}-{window.end}", 0.8, "E3"))
    return edges
