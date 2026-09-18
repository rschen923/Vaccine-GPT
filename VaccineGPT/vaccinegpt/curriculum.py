"""Deterministic curriculum scheduling and coupling policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_epoch: int
    end_epoch: int
    task_weights: Mapping[str, float] = field(default_factory=dict)
    detach_couplings: bool = True

    def __post_init__(self) -> None:
        if self.start_epoch < 0 or self.end_epoch <= self.start_epoch:
            raise ValueError("curriculum stage must have 0 <= start_epoch < end_epoch")
        if any(value < 0 for value in self.task_weights.values()):
            raise ValueError("curriculum weights cannot be negative")


class Curriculum:
    def __init__(self, stages: list[CurriculumStage] | None = None):
        self.stages = sorted(stages or [
            CurriculumStage("representation", 0, 2, {"T1": 1.0}, True),
            CurriculumStage("supervised", 2, 5, {"T1": 1.0, "T2": 0.5, "T3a": 0.5, "T3b": 0.5}, True),
            CurriculumStage("joint", 5, 10**9, {"T1": 1.0, "T2": 1.0, "T3a": 1.0, "T3b": 1.0, "T4": 1.0}, False),
        ], key=lambda stage: stage.start_epoch)
        for left, right in zip(self.stages, self.stages[1:]):
            if left.end_epoch > right.start_epoch:
                raise ValueError("curriculum stages overlap")

    def stage(self, epoch: int) -> CurriculumStage:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        for stage in self.stages:
            if stage.start_epoch <= epoch < stage.end_epoch:
                return stage
        return self.stages[-1]

