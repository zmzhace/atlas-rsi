"""Atlas-RSI: a bounded, auditable self-improvement control plane."""

from .core import (
    Candidate,
    Constitution,
    EvaluatorProposal,
    EvaluatorRegistry,
    EvolutionEngine,
    KnowledgeEntry,
    RoundReport,
    ScoreCard,
)
from .persistence import SQLiteExperimentStore
from .checkpoints import Checkpoint, CheckpointRegistry
from .domains import CommandEvolutionDomain
from .providers import GridProposer, OpenAICompatibleProposer
from .replay import ReplayPolicyOptimizer, ReplaySuggestion
from .runner import (
    CommandTaskRunner,
    DockerRunner,
    LocalProcessRunner,
    RunResult,
    TaskOutcome,
    TaskSpec,
)
from .stats import Estimate, conservative_delta, estimate

__all__ = [
    "Candidate",
    "Constitution",
    "EvaluatorProposal",
    "EvaluatorRegistry",
    "EvolutionEngine",
    "KnowledgeEntry",
    "RoundReport",
    "ScoreCard",
    "SQLiteExperimentStore",
    "Estimate",
    "conservative_delta",
    "estimate",
    "Checkpoint",
    "CheckpointRegistry",
    "CommandEvolutionDomain",
    "GridProposer",
    "OpenAICompatibleProposer",
    "ReplayPolicyOptimizer",
    "ReplaySuggestion",
    "CommandTaskRunner",
    "DockerRunner",
    "LocalProcessRunner",
    "RunResult",
    "TaskOutcome",
    "TaskSpec",
]
