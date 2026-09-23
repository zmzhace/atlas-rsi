from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from .core import EvolutionEngine, RoundReport
from .replay import ReplayPolicyOptimizer


def write_report(
    path: Path | str,
    engine: EvolutionEngine,
    reports: Sequence[RoundReport],
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    incumbent_score = engine.scores[engine.incumbent_id]
    replay = ReplayPolicyOptimizer().suggest(engine.archive, engine.scores)
    lines = [
        "# Atlas-RSI Experiment Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Constitution: `{engine.constitution.fingerprint()}`",
        f"- Active evaluator: `{engine.evaluators.active_version}`",
        f"- Final incumbent: `{engine.incumbent_id}`",
        f"- Holdout score: `{incumbent_score.holdout_score:.4f}`",
        f"- Safety score: `{incumbent_score.safety_score:.4f}`",
        f"- Candidates archived: `{len(engine.archive)}`",
        f"- Knowledge entries: `{len(engine.knowledge)}`",
        "",
        "## Round trajectory",
        "",
        "| Round | Before | Promoted | After | Rejected |",
        "|---:|---|---|---|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report.round_index} | `{report.incumbent_before}` | "
            f"`{report.promoted or 'none'}` | `{report.incumbent_after}` | "
            f"{len(report.rejected)} |"
        )
    lines.extend(["", "## Replay-derived policy suggestions", ""])
    if replay:
        lines.extend(
            [
                "| Payload key | Direction | Mean holdout delta | Observations |",
                "|---|---|---:|---:|",
            ]
        )
        for suggestion in replay[:20]:
            lines.append(
                f"| `{suggestion.key}` | {suggestion.preferred_direction} | "
                f"{suggestion.mean_observed_delta:.4f} | {suggestion.observations} |"
            )
    else:
        lines.append("No replay suggestion has enough evidence yet.")
    lines.extend(["", "## Dead ends", ""])
    dead_ends = [entry for entry in engine.knowledge if entry.kind == "dead_end"]
    if not dead_ends:
        lines.append("No rejected branch was recorded.")
    for entry in dead_ends[-30:]:
        reasons = entry.evidence.get("reasons", [])
        lines.append(
            f"- `{entry.candidate_id}`: {entry.statement} — "
            f"{', '.join(str(reason) for reason in reasons)}"
        )
    lines.extend(
        [
            "",
            "## Machine-readable final state",
            "",
            "```json",
            json.dumps(engine.snapshot(), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
