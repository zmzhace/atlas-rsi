from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .core import Constitution


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    path: str
    digest: str
    parent_id: str | None
    metrics: Mapping[str, float]
    anchors_passed: bool
    promoted: bool = False


class CheckpointRegistry:
    """Metadata-only registry. It never trains or loads weights by itself."""

    def __init__(self, registry_path: Path | str) -> None:
        self.registry_path = Path(registry_path)
        self.checkpoints: dict[str, Checkpoint] = {}
        self.active_id: str | None = None
        self._load()

    def register(
        self,
        checkpoint_id: str,
        path: Path | str,
        *,
        parent_id: str | None,
        metrics: Mapping[str, float],
        anchors_passed: bool,
    ) -> Checkpoint:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        digest = sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            path=str(checkpoint_path.resolve()),
            digest=digest,
            parent_id=parent_id,
            metrics=dict(metrics),
            anchors_passed=anchors_passed,
        )
        self.checkpoints[checkpoint_id] = checkpoint
        self._save()
        return checkpoint

    def promote(
        self,
        checkpoint_id: str,
        *,
        constitution: Constitution,
        minimum_metric: float,
        metric_name: str = "holdout",
        operator_approved: bool = False,
    ) -> bool:
        checkpoint = self.checkpoints[checkpoint_id]
        if not constitution.allow_weight_updates:
            return False
        if not checkpoint.anchors_passed:
            return False
        if checkpoint.metrics.get(metric_name, float("-inf")) < minimum_metric:
            return False
        if constitution.high_risk_requires_approval and not operator_approved:
            return False
        promoted = Checkpoint(**{**asdict(checkpoint), "promoted": True})
        self.checkpoints[checkpoint_id] = promoted
        self.active_id = checkpoint_id
        self._save()
        return True

    def verify(self, checkpoint_id: str) -> bool:
        checkpoint = self.checkpoints[checkpoint_id]
        path = Path(checkpoint.path)
        return path.is_file() and sha256(path.read_bytes()).hexdigest() == checkpoint.digest

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.active_id = payload.get("active_id")
        self.checkpoints = {
            key: Checkpoint(**value)
            for key, value in payload.get("checkpoints", {}).items()
        }

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_id": self.active_id,
            "checkpoints": {
                key: asdict(value) for key, value in self.checkpoints.items()
            },
        }
        self.registry_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
