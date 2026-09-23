from __future__ import annotations

import unittest
from typing import Sequence

from rsi_lab import (
    Candidate,
    Constitution,
    EvaluatorProposal,
    EvaluatorRegistry,
    EvolutionEngine,
    KnowledgeEntry,
    ScoreCard,
)


class FixedDomain:
    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self.candidates = candidates

    def propose(
        self,
        incumbent: Candidate,
        round_index: int,
        knowledge: Sequence[KnowledgeEntry],
        limit: int,
    ) -> Sequence[Candidate]:
        return self.candidates[:limit]

    def evaluate(self, candidate: Candidate, evaluator_version: str) -> ScoreCard:
        values = candidate.payload
        return ScoreCard(
            dev_score=float(values.get("dev", 0.5)),
            holdout_score=float(values.get("holdout", 0.5)),
            safety_score=float(values.get("safety", 1.0)),
            cost=float(values.get("cost", 1.0)),
            anchors_passed=bool(values.get("anchors", True)),
        )


def baseline() -> Candidate:
    return Candidate(
        candidate_id="base",
        parent_id=None,
        round_index=0,
        mutation_surface="harness",
        payload={"holdout": 0.5, "cost": 1.0},
        hypothesis="baseline",
    )


class EvolutionEngineTests(unittest.TestCase):
    def test_promotes_best_safe_holdout_candidate(self) -> None:
        candidates = [
            Candidate(
                "good",
                "base",
                1,
                "harness",
                {"holdout": 0.60, "cost": 1.1},
                "safe gain",
            ),
            Candidate(
                "better-dev-only",
                "base",
                1,
                "harness",
                {"dev": 0.9, "holdout": 0.505, "cost": 1.0},
                "overfits dev",
            ),
        ]
        engine = EvolutionEngine(
            Constitution(objective="test"), FixedDomain(candidates), baseline()
        )
        report = engine.run_round(1)
        self.assertEqual(report.promoted, "good")
        self.assertIn("insufficient_holdout_gain", report.rejected["better-dev-only"])

    def test_rejects_unsafe_or_disallowed_mutations(self) -> None:
        candidates = [
            Candidate(
                "unsafe",
                "base",
                1,
                "harness",
                {"holdout": 0.8, "safety": 0.0, "anchors": False},
                "unsafe gain",
            ),
            Candidate(
                "weights",
                "base",
                1,
                "weights",
                {"holdout": 0.9},
                "unauthorized training",
            ),
        ]
        engine = EvolutionEngine(
            Constitution(objective="test"), FixedDomain(candidates), baseline()
        )
        report = engine.run_round(1)
        self.assertIsNone(report.promoted)
        self.assertIn("anchor_failure", report.rejected["unsafe"])
        self.assertIn("mutation_surface_not_permitted", report.rejected["weights"])

    def test_rollback_restores_previous_incumbent(self) -> None:
        candidate = Candidate(
            "good",
            "base",
            1,
            "harness",
            {"holdout": 0.6, "cost": 1.0},
            "gain",
        )
        engine = EvolutionEngine(
            Constitution(objective="test"), FixedDomain([candidate]), baseline()
        )
        engine.run_round(1)
        self.assertEqual(engine.rollback(), "base")

    def test_evaluator_changes_only_at_guarded_boundary(self) -> None:
        registry = EvaluatorRegistry()
        proposal = EvaluatorProposal("evaluator-v2", 0.99, "better calibration")
        locked = Constitution(objective="test")
        enabled = Constitution(objective="test", allow_evaluator_updates=True)

        self.assertFalse(
            registry.try_promote(
                proposal,
                at_epoch_boundary=True,
                minimum_anchor_score=0.95,
                constitution=locked,
            )
        )
        self.assertFalse(
            registry.try_promote(
                proposal,
                at_epoch_boundary=False,
                minimum_anchor_score=0.95,
                constitution=enabled,
            )
        )
        self.assertTrue(
            registry.try_promote(
                proposal,
                at_epoch_boundary=True,
                minimum_anchor_score=0.95,
                constitution=enabled,
            )
        )


if __name__ == "__main__":
    unittest.main()
