from __future__ import annotations

from hashlib import sha256
from typing import Sequence

from .core import Candidate, KnowledgeEntry, ScoreCard
from .providers import CandidateProposer
from .runner import CommandTaskRunner, TaskOutcome, TaskSpec


class CommandEvolutionDomain:
    """Evolve a declarative harness against isolated command-based tasks."""

    def __init__(
        self,
        *,
        objective: str,
        proposer: CandidateProposer,
        task_runner: CommandTaskRunner,
        tasks: Sequence[TaskSpec],
    ) -> None:
        self.objective = objective
        self.proposer = proposer
        self.task_runner = task_runner
        self.tasks = tuple(tasks)
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("task ids must be unique")
        if not any(task.split == "dev" for task in self.tasks):
            raise ValueError("at least one development task is required")
        if not any(task.split == "holdout" for task in self.tasks):
            raise ValueError("at least one holdout task is required")
        if not any(task.split == "anchor" for task in self.tasks):
            raise ValueError("at least one safety anchor task is required")

    def propose(
        self,
        incumbent: Candidate,
        round_index: int,
        knowledge: Sequence[KnowledgeEntry],
        limit: int,
    ) -> Sequence[Candidate]:
        public_tasks = [
            {
                "task_id": task.task_id,
                "description": task.public_description,
            }
            for task in self.tasks
            if task.split == "dev"
        ]
        return self.proposer.propose(
            incumbent,
            round_index,
            knowledge,
            limit,
            {
                "objective": self.objective,
                "public_tasks": public_tasks,
            },
        )

    def evaluate(self, candidate: Candidate, evaluator_version: str) -> ScoreCard:
        outcomes: list[TaskOutcome] = []
        for task in self.tasks:
            for seed in task.seeds:
                outcomes.append(
                    self.task_runner.evaluate(task, candidate.payload, seed)
                )

        samples: dict[str, list[float]] = {"dev": [], "holdout": [], "anchor": []}
        weighted: dict[str, list[tuple[float, float]]] = {
            "dev": [],
            "holdout": [],
            "anchor": [],
        }
        elapsed = 0.0
        notes: list[str] = []
        task_weights = {task.task_id: task.weight for task in self.tasks}
        for outcome in outcomes:
            value = 1.0 if outcome.passed else 0.0
            samples[outcome.split].append(value)
            weighted[outcome.split].append((value, task_weights[outcome.task_id]))
            elapsed += outcome.solver.elapsed_seconds
            if outcome.evaluator:
                elapsed += outcome.evaluator.elapsed_seconds
            if outcome.reviewer:
                elapsed += outcome.reviewer.elapsed_seconds
            if not outcome.passed:
                notes.append(
                    f"{outcome.task_id}[seed={outcome.seed}] failed"
                )

        dev_score = self._weighted_mean(weighted["dev"], fallback=0.0)
        holdout_score = self._weighted_mean(weighted["holdout"])
        safety_score = self._weighted_mean(weighted["anchor"])
        return ScoreCard(
            dev_score=dev_score,
            holdout_score=holdout_score,
            safety_score=safety_score,
            cost=max(elapsed, 1e-6),
            anchors_passed=all(value == 1.0 for value in samples["anchor"]),
            metrics={
                "tasks": float(len(self.tasks)),
                "trials": float(len(outcomes)),
                "evaluator_version_hash": float(
                    int(sha256(evaluator_version.encode()).hexdigest()[:8], 16)
                ),
            },
            notes=tuple(notes[:20]),
            dev_samples=tuple(samples["dev"]),
            holdout_samples=tuple(samples["holdout"]),
        )

    @staticmethod
    def _weighted_mean(
        values: Sequence[tuple[float, float]], fallback: float | None = None
    ) -> float:
        if not values:
            if fallback is None:
                raise ValueError("required task split has no outcomes")
            return fallback
        denominator = sum(weight for _, weight in values)
        return sum(value * weight for value, weight in values) / denominator
