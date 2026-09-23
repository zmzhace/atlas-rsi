from __future__ import annotations

from typing import Sequence

from rsi_lab import (
    Candidate,
    Constitution,
    EvolutionEngine,
    KnowledgeEntry,
    ScoreCard,
)


class HarnessDemoDomain:
    """Deterministic demo: evolve safe, declarative agent-harness parameters."""

    def propose(
        self,
        incumbent: Candidate,
        round_index: int,
        knowledge: Sequence[KnowledgeEntry],
        limit: int,
    ) -> Sequence[Candidate]:
        current = dict(incumbent.payload)
        proposals = [
            (
                "verification",
                {**current, "verification_depth": current["verification_depth"] + 1},
                "More verification should reduce silent failures.",
            ),
            (
                "context",
                {**current, "context_budget": current["context_budget"] + 1},
                "A slightly larger context budget should improve task completion.",
            ),
            (
                "retries",
                {**current, "retry_limit": current["retry_limit"] + 3},
                "Aggressive retries may recover more failures.",
            ),
            (
                "unsafe",
                {**current, "skip_safety_checks": True},
                "Skipping safety checks would make the demo faster.",
            ),
        ]
        return [
            Candidate(
                candidate_id=f"r{round_index}-{name}",
                parent_id=incumbent.candidate_id,
                round_index=round_index,
                mutation_surface="harness",
                payload=payload,
                hypothesis=hypothesis,
            )
            for name, payload, hypothesis in proposals[:limit]
        ]

    def evaluate(self, candidate: Candidate, evaluator_version: str) -> ScoreCard:
        payload = candidate.payload
        retries = float(payload.get("retry_limit", 0))
        context = float(payload.get("context_budget", 0))
        verification = float(payload.get("verification_depth", 0))
        unsafe = bool(payload.get("skip_safety_checks", False))

        holdout = 0.50 + 0.025 * min(context, 4) + 0.04 * min(verification, 3)
        holdout += 0.005 * min(retries, 2)
        dev = holdout + 0.03 * max(retries - 2, 0)
        cost = 1.0 + 0.04 * context + 0.06 * verification + 0.05 * retries
        return ScoreCard(
            dev_score=round(dev, 4),
            holdout_score=round(holdout, 4),
            safety_score=0.0 if unsafe else 1.0,
            cost=round(cost, 4),
            anchors_passed=not unsafe,
            metrics={"evaluator_version": 1.0 if evaluator_version else 0.0},
        )


def main() -> None:
    constitution = Constitution(
        objective="Improve held-out coding-agent success without safety regression",
        max_rounds=3,
        minimum_holdout_delta=0.01,
        maximum_cost_ratio=1.25,
    )
    baseline = Candidate(
        candidate_id="baseline",
        parent_id=None,
        round_index=0,
        mutation_surface="harness",
        payload={"retry_limit": 1, "context_budget": 1, "verification_depth": 1},
        hypothesis="Operator-audited baseline",
        proposer="operator",
    )
    engine = EvolutionEngine(constitution, HarnessDemoDomain(), baseline)

    for round_index in range(1, constitution.max_rounds + 1):
        report = engine.run_round(round_index)
        print(
            f"round={round_index} promoted={report.promoted} "
            f"incumbent={report.incumbent_after}"
        )

    final_score = engine.scores[engine.incumbent_id]
    print(f"final={engine.incumbent_id} holdout={final_score.holdout_score}")
    print(f"knowledge_entries={len(engine.knowledge)}")


if __name__ == "__main__":
    main()
