from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .core import Candidate, ScoreCard


@dataclass(frozen=True)
class ReplaySuggestion:
    key: str
    preferred_direction: str
    mean_observed_delta: float
    observations: int


class ReplayPolicyOptimizer:
    """Dream-RSI-style offline analysis over already observed candidate branches."""

    def suggest(
        self,
        archive: Mapping[str, Candidate],
        scores: Mapping[str, ScoreCard],
        *,
        minimum_observations: int = 1,
    ) -> Sequence[ReplaySuggestion]:
        changes: dict[tuple[str, str], list[float]] = {}
        for candidate_id, candidate in archive.items():
            if not candidate.parent_id or candidate_id not in scores:
                continue
            if candidate.parent_id not in archive or candidate.parent_id not in scores:
                continue
            parent = archive[candidate.parent_id]
            candidate_score = scores[candidate_id]
            parent_score = scores[candidate.parent_id]
            if (
                not candidate_score.anchors_passed
                or candidate_score.safety_score < parent_score.safety_score
            ):
                continue
            score_delta = (
                candidate_score.holdout_score - parent_score.holdout_score
            )
            keys = set(parent.payload) | set(candidate.payload)
            for key in keys:
                before = parent.payload.get(key)
                after = candidate.payload.get(key)
                if before == after:
                    continue
                direction = self._direction(before, after)
                changes.setdefault((key, direction), []).append(score_delta)

        suggestions: list[ReplaySuggestion] = []
        for (key, direction), deltas in changes.items():
            if len(deltas) < minimum_observations:
                continue
            mean_delta = sum(deltas) / len(deltas)
            if mean_delta <= 0.0:
                continue
            suggestions.append(
                ReplaySuggestion(
                    key=key,
                    preferred_direction=direction,
                    mean_observed_delta=mean_delta,
                    observations=len(deltas),
                )
            )
        return tuple(
            sorted(
                suggestions,
                key=lambda item: (item.mean_observed_delta, item.observations),
                reverse=True,
            )
        )

    @staticmethod
    def _direction(before: Any, after: Any) -> str:
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            return "increase" if after > before else "decrease"
        if before is None:
            return "add"
        if after is None:
            return "remove"
        return "replace"
