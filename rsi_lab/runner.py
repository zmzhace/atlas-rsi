from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RunResult:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False


class Runner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> RunResult: ...


class LocalProcessRunner:
    """Local runner for trusted commands and tests. Prefer Docker for generated code."""

    def __init__(self, *, max_output_chars: int = 20_000) -> None:
        self.max_output_chars = max_output_chars

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> RunResult:
        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            **dict(env),
        }
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            return RunResult(
                command=tuple(command),
                exit_code=completed.returncode,
                stdout=completed.stdout[-self.max_output_chars :],
                stderr=completed.stderr[-self.max_output_chars :],
                elapsed_seconds=elapsed,
            )
        except subprocess.TimeoutExpired as error:
            elapsed = time.monotonic() - started
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return RunResult(
                command=tuple(command),
                exit_code=124,
                stdout=stdout[-self.max_output_chars :],
                stderr=stderr[-self.max_output_chars :],
                elapsed_seconds=elapsed,
                timed_out=True,
            )


class DockerRunner:
    """Hardened Docker runner with no network, dropped capabilities and limits."""

    def __init__(
        self,
        image: str,
        *,
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 256,
        max_output_chars: int = 20_000,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.local = LocalProcessRunner(max_output_chars=max_output_chars)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> RunResult:
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "-v",
            f"{cwd.resolve()}:/workspace:rw",
            "-w",
            "/workspace",
        ]
        for key, value in sorted(env.items()):
            docker_command.extend(["-e", f"{key}={value}"])
        docker_command.append(self.image)
        docker_command.extend(command)
        return self.local.run(
            docker_command,
            cwd=cwd,
            env={},
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    public_dir: Path
    test_command: tuple[str, ...]
    public_description: str
    private_evaluator_dir: Path | None = None
    review_command: tuple[str, ...] | None = None
    seeds: tuple[int, ...] = (0, 1, 2)
    timeout_seconds: float = 60.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.split not in {"dev", "holdout", "anchor"}:
            raise ValueError(f"unsupported split: {self.split}")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.timeout_seconds <= 0.0 or self.weight <= 0.0:
            raise ValueError("task timeout and weight must be positive")


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    split: str
    seed: int
    passed: bool
    solver: RunResult
    evaluator: RunResult | None
    reviewer: RunResult | None = None


class CommandTaskRunner:
    """Run an external solver, then reveal and execute private evaluator files."""

    def __init__(
        self,
        runner: Runner,
        solver_command: Sequence[str],
    ) -> None:
        self.runner = runner
        self.solver_command = tuple(solver_command)

    def evaluate(
        self,
        task: TaskSpec,
        candidate_payload: Mapping[str, object],
        seed: int,
    ) -> TaskOutcome:
        with tempfile.TemporaryDirectory(prefix="atlas-rsi-") as temporary:
            workspace = Path(temporary) / "workspace"
            shutil.copytree(task.public_dir, workspace)
            harness_path = workspace / ".atlas-harness.json"
            harness_path.write_text(
                json.dumps(dict(candidate_payload), sort_keys=True),
                encoding="utf-8",
            )
            environment = {
                "ATLAS_TASK_ID": task.task_id,
                "ATLAS_SEED": str(seed),
                "ATLAS_HARNESS_PATH": "/workspace/.atlas-harness.json"
                if isinstance(self.runner, DockerRunner)
                else str(harness_path),
                "PYTHONPATH": "/workspace"
                if isinstance(self.runner, DockerRunner)
                else str(workspace),
            }
            solver_result = self.runner.run(
                self.solver_command,
                cwd=workspace,
                env=environment,
                timeout_seconds=task.timeout_seconds,
            )
            if solver_result.exit_code != 0:
                return TaskOutcome(
                    task.task_id,
                    task.split,
                    seed,
                    False,
                    solver_result,
                    None,
                    None,
                )

            if task.private_evaluator_dir is not None:
                private_target = workspace / ".atlas-private"
                shutil.copytree(task.private_evaluator_dir, private_target)
            evaluator_result = self.runner.run(
                task.test_command,
                cwd=workspace,
                env=environment,
                timeout_seconds=task.timeout_seconds,
            )
            reviewer_result = None
            if task.review_command is not None:
                reviewer_result = self.runner.run(
                    task.review_command,
                    cwd=workspace,
                    env=environment,
                    timeout_seconds=task.timeout_seconds,
                )
            return TaskOutcome(
                task.task_id,
                task.split,
                seed,
                evaluator_result.exit_code == 0
                and (reviewer_result is None or reviewer_result.exit_code == 0),
                solver_result,
                evaluator_result,
                reviewer_result,
            )
