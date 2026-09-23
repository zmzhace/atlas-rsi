from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist, mean, stdev
from typing import Sequence


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float
    n: int


def estimate(samples: Sequence[float], confidence: float = 0.95) -> Estimate:
    """Return a Wilson interval for binary samples, otherwise a normal interval."""

    values = tuple(float(value) for value in samples)
    if not values:
        raise ValueError("at least one sample is required")
    center = mean(values)
    if len(values) == 1:
        return Estimate(center, center, center, 1)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    if all(value in {0.0, 1.0} for value in values):
        n = len(values)
        denominator = 1.0 + z_score**2 / n
        adjusted = (center + z_score**2 / (2.0 * n)) / denominator
        margin = (
            z_score
            * sqrt(center * (1.0 - center) / n + z_score**2 / (4.0 * n**2))
            / denominator
        )
        return Estimate(center, max(0.0, adjusted - margin), min(1.0, adjusted + margin), n)
    margin = z_score * stdev(values) / sqrt(len(values))
    return Estimate(center, center - margin, center + margin, len(values))


def conservative_delta(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    confidence: float = 0.95,
) -> float:
    """Candidate lower bound minus incumbent upper bound."""

    candidate_estimate = estimate(candidate, confidence)
    incumbent_estimate = estimate(incumbent, confidence)
    return candidate_estimate.lower - incumbent_estimate.upper
