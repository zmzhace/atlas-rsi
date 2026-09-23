from __future__ import annotations

import argparse
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .core import Candidate, Constitution, EvolutionEngine
from .domains import CommandEvolutionDomain
from .persistence import SQLiteExperimentStore
from .providers import GridProposer, OpenAICompatibleProposer
from .reporting import write_report
from .runner import CommandTaskRunner, DockerRunner, LocalProcessRunner, TaskSpec


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_command(base: Path, command: Sequence[str]) -> tuple[str, ...]:
    resolved: list[str] = []
    for index, token in enumerate(command):
        candidate = base / token
        if not Path(token).is_absolute() and candidate.exists():
            resolved.append(str(candidate.resolve()))
        else:
            resolved.append(token)
    return tuple(resolved)


def _constitution(raw: Mapping[str, Any]) -> Constitution:
    allowed = {item.name for item in fields(Constitution)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown constitution fields: {sorted(unknown)}")
    payload = dict(raw)
    if "allowed_surfaces" in payload:
        payload["allowed_surfaces"] = tuple(payload["allowed_surfaces"])
    return Constitution(**payload)


def _experiment_fingerprint(config: Mapping[str, Any]) -> str:
    relevant = {
        key: config[key]
        for key in (
            "constitution", "baseline", "proposer", "runner",
            "solver_command", "tasks",
        )
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _build_domain(config: Mapping[str, Any], base: Path) -> CommandEvolutionDomain:
    proposer_config = config["proposer"]
    proposer_type = proposer_config.get("type", "grid")
    if proposer_type == "grid":
        proposer = GridProposer(proposer_config["mutations"])
    elif proposer_type == "openai-compatible":
        proposer = OpenAICompatibleProposer(
            model=proposer_config["model"],
            base_url=proposer_config.get("base_url", "https://api.openai.com/v1"),
            api_key_env=proposer_config.get("api_key_env", "OPENAI_API_KEY"),
            timeout_seconds=float(proposer_config.get("timeout_seconds", 90)),
            allowed_surfaces=proposer_config.get(
                "allowed_surfaces", ["harness", "search_policy", "knowledge"]
            ),
        )
    else:
        raise ValueError(f"unknown proposer type: {proposer_type}")

    runner_config = config["runner"]
    runner_type = runner_config.get("type", "docker")
    if runner_type == "local":
        runner = LocalProcessRunner()
    elif runner_type == "docker":
        runner = DockerRunner(
            runner_config["image"],
            memory=runner_config.get("memory", "2g"),
            cpus=str(runner_config.get("cpus", "2")),
            pids_limit=int(runner_config.get("pids_limit", 256)),
        )
    else:
        raise ValueError(f"unknown runner type: {runner_type}")

    solver_command = _resolve_command(base, config["solver_command"])
    task_runner = CommandTaskRunner(runner, solver_command)
    tasks = []
    for raw in config["tasks"]:
        private_dir = raw.get("private_evaluator_dir")
        tasks.append(
            TaskSpec(
                task_id=raw["task_id"],
                split=raw["split"],
                public_dir=_resolve_path(base, raw["public_dir"]),
                private_evaluator_dir=(
                    _resolve_path(base, private_dir) if private_dir else None
                ),
                test_command=tuple(raw["test_command"]),
                review_command=(
                    tuple(raw["review_command"])
                    if raw.get("review_command")
                    else None
                ),
                public_description=raw["public_description"],
                seeds=tuple(int(seed) for seed in raw.get("seeds", [0, 1, 2])),
                timeout_seconds=float(raw.get("timeout_seconds", 60)),
                weight=float(raw.get("weight", 1.0)),
            )
        )
    return CommandEvolutionDomain(
        objective=config["constitution"]["objective"],
        proposer=proposer,
        task_runner=task_runner,
        tasks=tasks,
    )


def run_experiment(config_path: Path) -> int:
    config_path = config_path.resolve()
    base = config_path.parent.parent if config_path.parent.name == "examples" else config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    constitution = _constitution(config["constitution"])
    domain = _build_domain(config, base)
    state_path = _resolve_path(base, config.get("state_path", "runs/state.sqlite3"))
    report_path = _resolve_path(base, config.get("report_path", "runs/report.md"))
    store = SQLiteExperimentStore(state_path)
    experiment_fingerprint = _experiment_fingerprint(config)

    existing_snapshot = store.latest_snapshot()
    resume = bool(config.get("resume", False))
    if not resume and (existing_snapshot is not None or store.events()):
        raise ValueError(
            "state database already contains an experiment; "
            "set resume=true or choose a new state_path"
        )
    if resume and existing_snapshot is None and store.events():
        raise ValueError("state database has no recoverable snapshot")
    snapshot = existing_snapshot if resume else None
    if snapshot:
        stored_fingerprint = snapshot.get("experiment_fingerprint")
        if stored_fingerprint and stored_fingerprint != experiment_fingerprint:
            raise ValueError("experiment configuration changed since the snapshot")
        engine = EvolutionEngine.restore(snapshot, domain, audit_log=store)
        next_round = engine.completed_rounds + 1
    else:
        raw_baseline = config["baseline"]
        baseline = Candidate(
            candidate_id=raw_baseline.get("candidate_id", "baseline"),
            parent_id=None,
            round_index=0,
            mutation_surface=raw_baseline.get("mutation_surface", "harness"),
            payload=raw_baseline["payload"],
            hypothesis=raw_baseline.get("hypothesis", "operator baseline"),
            proposer="operator",
        )
        engine = EvolutionEngine(
            constitution,
            domain,
            baseline,
            audit_log=store,
        )
        next_round = 1

    for round_index in range(next_round, constitution.max_rounds + 1):
        report = engine.run_round(round_index)
        snapshot_payload = engine.snapshot()
        snapshot_payload["experiment_fingerprint"] = experiment_fingerprint
        store.save_snapshot(round_index, snapshot_payload)
        print(
            f"round={round_index} promoted={report.promoted or 'none'} "
            f"incumbent={report.incumbent_after}"
        )
    output = write_report(report_path, engine, engine.round_reports)
    print(f"report={output}")
    print(f"state={state_path}")
    return 0


def inspect_store(path: Path) -> int:
    store = SQLiteExperimentStore(path)
    snapshot = store.latest_snapshot()
    payload = {
        "events": len(store.events()),
        "latest_snapshot": snapshot,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas-rsi",
        description="Bounded and auditable RSI experiment control plane",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run an experiment JSON")
    run_parser.add_argument("config", type=Path)
    inspect_parser = subparsers.add_parser("inspect", help="inspect a state database")
    inspect_parser.add_argument("state", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "run":
            return run_experiment(arguments.config)
        if arguments.command == "inspect":
            return inspect_store(arguments.state)
    except (KeyError, ValueError, RuntimeError, OSError) as error:
        print(f"atlas-rsi: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
