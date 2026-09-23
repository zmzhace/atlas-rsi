from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .stats import conservative_delta


@dataclass(frozen=True)
class Constitution:
    """Operator-owned invariants. Candidates never receive a write handle to this."""

    objective: str
    allowed_surfaces: tuple[str, ...] = (
        "knowledge",
        "harness",
        "search_policy",
    )
    max_candidates_per_round: int = 8
    max_rounds: int = 4
    minimum_holdout_delta: float = 0.01
    minimum_safety_score: float = 1.0
    maximum_cost_ratio: float = 1.25
    confidence_level: float = 0.95
    minimum_trials_for_ci: int = 10
    allow_weight_updates: bool = False
    allow_evaluator_updates: bool = False
    high_risk_requires_approval: bool = True

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("objective must not be empty")
        if self.max_candidates_per_round < 1 or self.max_rounds < 1:
            raise ValueError("candidate and round budgets must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if self.minimum_trials_for_ci < 2:
            raise ValueError("minimum_trials_for_ci must be at least two")
        if self.maximum_cost_ratio <= 0.0:
            raise ValueError("maximum_cost_ratio must be positive")

    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest()[:16]

    def permits(self, surface: str) -> bool:
        if surface == "weights":
            return self.allow_weight_updates
        if surface == "evaluator":
            return self.allow_evaluator_updates
        return surface in self.allowed_surfaces


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parent_id: str | None
    round_index: int
    mutation_surface: str
    payload: Mapping[str, Any]
    hypothesis: str
    proposer: str = "worker"


@dataclass(frozen=True)
class ScoreCard:
    dev_score: float
    holdout_score: float
    safety_score: float
    cost: float
    anchors_passed: bool
    metrics: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    dev_samples: tuple[float, ...] = ()
    holdout_samples: tuple[float, ...] = ()


@dataclass(frozen=True)
class KnowledgeEntry:
    round_index: int
    candidate_id: str
    kind: str
    statement: str
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class RoundReport:
    round_index: int
    incumbent_before: str
    incumbent_after: str
    promoted: str | None
    rejected: Mapping[str, tuple[str, ...]]
    scores: Mapping[str, ScoreCard]
    constitution_fingerprint: str


@dataclass(frozen=True)
class EvaluatorProposal:
    version: str
    anchor_score: float
    rationale: str


class Domain(Protocol):
    """A real implementation should evaluate inside an isolated runner."""

    def propose(
        self,
        incumbent: Candidate,
        round_index: int,
        knowledge: Sequence[KnowledgeEntry],
        limit: int,
    ) -> Sequence[Candidate]: ...

    def evaluate(self, candidate: Candidate, evaluator_version: str) -> ScoreCard: ...


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []

    def append(self, event: str, payload: Mapping[str, Any]) -> None:
        record = {"event": event, **payload}
        self.events.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


class EvaluatorRegistry:
    """RQGM-style slow evaluator evolution guarded by immutable anchors."""

    def __init__(self, initial_version: str = "evaluator-v1") -> None:
        self.active_version = initial_version
        self.history = [initial_version]

    def try_promote(
        self,
        proposal: EvaluatorProposal,
        *,
        at_epoch_boundary: bool,
        minimum_anchor_score: float,
        constitution: Constitution,
        operator_approved: bool = False,
    ) -> bool:
        if not constitution.allow_evaluator_updates:
            return False
        if not at_epoch_boundary:
            return False
        if proposal.anchor_score < minimum_anchor_score:
            return False
        if constitution.high_risk_requires_approval and not operator_approved:
            return False
        self.active_version = proposal.version
        self.history.append(proposal.version)
        return True


class EvolutionEngine:
    """Coordinates bounded candidate generation, evaluation, promotion and rollback."""

    def __init__(
        self,
        constitution: Constitution,
        domain: Domain,
        baseline: Candidate,
        *,
        evaluator_registry: EvaluatorRegistry | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.constitution = constitution
        self.domain = domain
        self.evaluators = evaluator_registry or EvaluatorRegistry()
        self.audit = audit_log or AuditLog()
        self.archive: dict[str, Candidate] = {baseline.candidate_id: baseline}
        self.scores: dict[str, ScoreCard] = {}
        self.knowledge: list[KnowledgeEntry] = []
        self.incumbent_id = baseline.candidate_id
        self.promotion_history = [baseline.candidate_id]
        self.completed_rounds = 0
        self.round_reports: list[RoundReport] = []

        baseline_score = self.domain.evaluate(
            baseline, self.evaluators.active_version
        )
        self._validate_scorecard(baseline_score)
        if (
            not baseline_score.anchors_passed
            or baseline_score.safety_score < constitution.minimum_safety_score
        ):
            raise ValueError("baseline must pass all safety gates")
        self.scores[baseline.candidate_id] = baseline_score
        self.audit.append(
            "baseline_registered",
            {
                "candidate_id": baseline.candidate_id,
                "score": asdict(baseline_score),
                "constitution": constitution.fingerprint(),
            },
        )

    @property
    def incumbent(self) -> Candidate:
        return self.archive[self.incumbent_id]

    def run_round(self, round_index: int) -> RoundReport:
        if round_index < 1 or round_index > self.constitution.max_rounds:
            raise ValueError("round index is outside the constitutional budget")
        if round_index != self.completed_rounds + 1:
            raise ValueError("rounds must run exactly once and in order")

        incumbent_before = self.incumbent
        incumbent_score = self.scores[incumbent_before.candidate_id]
        proposed = list(
            self.domain.propose(
                incumbent_before,
                round_index,
                tuple(self.knowledge),
                self.constitution.max_candidates_per_round,
            )
        )
        if len(proposed) > self.constitution.max_candidates_per_round:
            raise ValueError("domain proposed more candidates than the round budget")

        eligible: list[tuple[Candidate, ScoreCard]] = []
        rejected: dict[str, tuple[str, ...]] = {}
        round_scores: dict[str, ScoreCard] = {}

        for candidate in proposed:
            reasons = self._validate_candidate(candidate, round_index)
            if reasons:
                rejected[candidate.candidate_id] = tuple(reasons)
                self._remember_rejection(candidate, reasons, None)
                continue

            score = self.domain.evaluate(candidate, self.evaluators.active_version)
            self._validate_scorecard(score)
            self.archive[candidate.candidate_id] = candidate
            self.scores[candidate.candidate_id] = score
            round_scores[candidate.candidate_id] = score
            reasons = self._promotion_failures(score, incumbent_score)
            if reasons:
                rejected[candidate.candidate_id] = tuple(reasons)
                self._remember_rejection(candidate, reasons, score)
            else:
                eligible.append((candidate, score))

        promoted: Candidate | None = None
        if eligible:
            promoted, promoted_score = max(
                eligible,
                key=lambda item: (
                    item[1].holdout_score,
                    item[1].safety_score,
                    -item[1].cost,
                ),
            )
            self.incumbent_id = promoted.candidate_id
            self.promotion_history.append(promoted.candidate_id)
            self.knowledge.append(
                KnowledgeEntry(
                    round_index=round_index,
                    candidate_id=promoted.candidate_id,
                    kind="promoted_lesson",
                    statement=promoted.hypothesis,
                    evidence={
                        "holdout_delta": (
                            promoted_score.holdout_score
                            - incumbent_score.holdout_score
                        ),
                        "safety": promoted_score.safety_score,
                        "cost": promoted_score.cost,
                    },
                )
            )

        report = RoundReport(
            round_index=round_index,
            incumbent_before=incumbent_before.candidate_id,
            incumbent_after=self.incumbent_id,
            promoted=promoted.candidate_id if promoted else None,
            rejected=rejected,
            scores=round_scores,
            constitution_fingerprint=self.constitution.fingerprint(),
        )
        self.completed_rounds = round_index
        self.round_reports.append(report)
        self.audit.append("round_completed", self._report_dict(report))
        return report

    def rollback(self) -> str:
        if len(self.promotion_history) == 1:
            return self.incumbent_id
        rolled_back = self.promotion_history.pop()
        self.incumbent_id = self.promotion_history[-1]
        self.audit.append(
            "rollback",
            {"from": rolled_back, "to": self.incumbent_id},
        )
        return self.incumbent_id

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 2,
            "constitution": asdict(self.constitution),
            "archive": {
                key: asdict(value) for key, value in self.archive.items()
            },
            "scores": {key: asdict(value) for key, value in self.scores.items()},
            "knowledge": [asdict(entry) for entry in self.knowledge],
            "incumbent_id": self.incumbent_id,
            "promotion_history": list(self.promotion_history),
            "completed_rounds": self.completed_rounds,
            "round_reports": [
                self._report_dict(report) for report in self.round_reports
            ],
            "evaluator_history": list(self.evaluators.history),
            "active_evaluator": self.evaluators.active_version,
        }

    @classmethod
    def restore(
        cls,
        snapshot: Mapping[str, Any],
        domain: Domain,
        *,
        audit_log: AuditLog | None = None,
    ) -> "EvolutionEngine":
        if snapshot.get("version") not in {1, 2}:
            raise ValueError("unsupported snapshot version")
        engine = cls.__new__(cls)
        constitution_payload = dict(snapshot["constitution"])
        constitution_payload["allowed_surfaces"] = tuple(
            constitution_payload["allowed_surfaces"]
        )
        engine.constitution = Constitution(**constitution_payload)
        engine.domain = domain
        engine.evaluators = EvaluatorRegistry(snapshot["evaluator_history"][0])
        engine.evaluators.history = list(snapshot["evaluator_history"])
        engine.evaluators.active_version = snapshot["active_evaluator"]
        engine.audit = audit_log or AuditLog()
        engine.archive = {
            key: Candidate(**value) for key, value in snapshot["archive"].items()
        }
        engine.scores = {
            key: cls._restore_score(value)
            for key, value in snapshot["scores"].items()
        }
        engine.knowledge = [
            KnowledgeEntry(**value) for value in snapshot["knowledge"]
        ]
        engine.incumbent_id = snapshot["incumbent_id"]
        engine.promotion_history = list(snapshot["promotion_history"])
        legacy_rounds = max(
            (entry.round_index for entry in engine.knowledge), default=0
        )
        engine.completed_rounds = int(
            snapshot.get(
                "completed_rounds",
                max(len(snapshot.get("round_reports", [])), legacy_rounds),
            )
        )
        engine.round_reports = [
            RoundReport(
                round_index=int(value["round_index"]),
                incumbent_before=value["incumbent_before"],
                incumbent_after=value["incumbent_after"],
                promoted=value.get("promoted"),
                rejected={
                    key: tuple(reasons)
                    for key, reasons in value["rejected"].items()
                },
                scores={
                    key: cls._restore_score(score)
                    for key, score in value["scores"].items()
                },
                constitution_fingerprint=value["constitution_fingerprint"],
            )
            for value in snapshot.get("round_reports", [])
        ]
        engine.audit.append(
            "snapshot_restored",
            {"incumbent_id": engine.incumbent_id},
        )
        return engine

    @staticmethod
    def _restore_score(value: Mapping[str, Any]) -> ScoreCard:
        payload = dict(value)
        payload["notes"] = tuple(payload.get("notes", ()))
        payload["dev_samples"] = tuple(payload.get("dev_samples", ()))
        payload["holdout_samples"] = tuple(payload.get("holdout_samples", ()))
        return ScoreCard(**payload)

    def _validate_candidate(self, candidate: Candidate, round_index: int) -> list[str]:
        reasons: list[str] = []
        if candidate.round_index != round_index:
            reasons.append("round_mismatch")
        if candidate.parent_id != self.incumbent_id:
            reasons.append("candidate_did_not_fork_from_incumbent")
        if not self.constitution.permits(candidate.mutation_surface):
            reasons.append("mutation_surface_not_permitted")
        if candidate.candidate_id in self.archive:
            reasons.append("duplicate_candidate_id")
        return reasons

    @staticmethod
    def _validate_scorecard(score: ScoreCard) -> None:
        bounded = (score.dev_score, score.holdout_score, score.safety_score)
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("scores must be finite values between zero and one")
        if not isfinite(score.cost) or score.cost <= 0.0:
            raise ValueError("evaluation cost must be finite and positive")
        samples = (*score.dev_samples, *score.holdout_samples)
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in samples):
            raise ValueError("score samples must be finite values between zero and one")

    def _promotion_failures(
        self, score: ScoreCard, incumbent: ScoreCard
    ) -> list[str]:
        reasons: list[str] = []
        if not score.anchors_passed:
            reasons.append("anchor_failure")
        if score.safety_score < self.constitution.minimum_safety_score:
            reasons.append("safety_regression")
        delta = score.holdout_score - incumbent.holdout_score
        enough_samples = (
            len(score.holdout_samples) >= self.constitution.minimum_trials_for_ci
            and len(incumbent.holdout_samples)
            >= self.constitution.minimum_trials_for_ci
        )
        if enough_samples:
            delta = conservative_delta(
                score.holdout_samples,
                incumbent.holdout_samples,
                self.constitution.confidence_level,
            )
        if delta < self.constitution.minimum_holdout_delta:
            reasons.append("insufficient_holdout_gain")
        baseline_cost = max(incumbent.cost, 1e-12)
        if score.cost / baseline_cost > self.constitution.maximum_cost_ratio:
            reasons.append("cost_budget_exceeded")
        return reasons

    def _remember_rejection(
        self,
        candidate: Candidate,
        reasons: Sequence[str],
        score: ScoreCard | None,
    ) -> None:
        self.knowledge.append(
            KnowledgeEntry(
                round_index=candidate.round_index,
                candidate_id=candidate.candidate_id,
                kind="dead_end",
                statement=candidate.hypothesis,
                evidence={
                    "reasons": list(reasons),
                    "score": asdict(score) if score else None,
                },
            )
        )

    @staticmethod
    def _report_dict(report: RoundReport) -> dict[str, Any]:
        return {
            "round_index": report.round_index,
            "incumbent_before": report.incumbent_before,
            "incumbent_after": report.incumbent_after,
            "promoted": report.promoted,
            "rejected": report.rejected,
            "scores": {
                key: asdict(value) for key, value in report.scores.items()
            },
            "constitution_fingerprint": report.constitution_fingerprint,
        }
