from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from rsi_lab import (
    Candidate,
    CheckpointRegistry,
    CommandTaskRunner,
    Constitution,
    DockerRunner,
    EvolutionEngine,
    GridProposer,
    LocalProcessRunner,
    OpenAICompatibleProposer,
    ReplayPolicyOptimizer,
    RunResult,
    ScoreCard,
    SQLiteExperimentStore,
    TaskSpec,
    conservative_delta,
    estimate,
)
from rsi_lab.cli import run_experiment


class RoundDomain:
    def propose(self, incumbent, round_index, knowledge, limit):
        if round_index == 1:
            return [
                Candidate(
                    "no-gain", incumbent.candidate_id, 1, "harness",
                    {"holdout": 0.5}, "no gain"
                )
            ]
        return [
            Candidate(
                "gain", incumbent.candidate_id, 2, "harness",
                {"holdout": 0.9}, "gain"
            )
        ]

    def evaluate(self, candidate, evaluator_version):
        value = float(candidate.payload.get("holdout", 0.5))
        return ScoreCard(value, value, 1.0, 1.0, True)


class StatisticsAndStateTests(unittest.TestCase):
    def test_estimates_and_conservative_delta(self) -> None:
        interval = estimate([1.0, 1.0, 1.0])
        self.assertLess(interval.lower, 1.0)
        self.assertEqual(interval.upper, 1.0)
        self.assertGreater(
            conservative_delta([1] * 100, [0] * 100), 0.8
        )
        with self.assertRaises(ValueError):
            estimate([])
        with self.assertRaises(ValueError):
            Constitution(objective="", confidence_level=2.0)

    def test_resume_after_non_promotion_runs_next_round(self) -> None:
        baseline = Candidate("base", None, 0, "harness", {"holdout": 0.5}, "base")
        engine = EvolutionEngine(
            Constitution(objective="resume", max_rounds=2), RoundDomain(), baseline
        )
        first = engine.run_round(1)
        self.assertIsNone(first.promoted)
        restored = EvolutionEngine.restore(
            json.loads(json.dumps(engine.snapshot())), RoundDomain()
        )
        self.assertEqual(restored.completed_rounds, 1)
        self.assertEqual(len(restored.round_reports), 1)
        second = restored.run_round(2)
        self.assertEqual(second.promoted, "gain")
        with self.assertRaises(ValueError):
            restored.run_round(2)

    def test_sqlite_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteExperimentStore(Path(directory) / "state.sqlite3")
            store.append("test", {"ok": True})
            store.save_snapshot(1, {"version": 2, "answer": 42})
            self.assertEqual(store.events()[0]["payload"], {"ok": True})
            self.assertEqual(store.latest_snapshot()["answer"], 42)


class ProviderAndReplayTests(unittest.TestCase):
    def test_grid_and_llm_payloads_extend_incumbent(self) -> None:
        incumbent = Candidate("base", None, 0, "harness", {"a": 1}, "base")
        grid = GridProposer([{"payload": {"b": 2}}])
        candidate = grid.propose(incumbent, 1, (), 1, {})[0]
        self.assertEqual(candidate.payload, {"a": 1, "b": 2})
        parsed = OpenAICompatibleProposer._parse_json_array(
            "```json\n[{\"payload\": {\"b\": 2}}]\n```"
        )
        self.assertEqual(parsed[0]["payload"], {"b": 2})

    def test_replay_finds_positive_direction(self) -> None:
        base = Candidate("base", None, 0, "harness", {"depth": 1}, "base")
        child = Candidate("child", "base", 1, "harness", {"depth": 2}, "deeper")
        scores = {
            "base": ScoreCard(0.5, 0.5, 1.0, 1.0, True),
            "child": ScoreCard(0.7, 0.8, 1.0, 1.0, True),
        }
        suggestion = ReplayPolicyOptimizer().suggest(
            {"base": base, "child": child}, scores
        )[0]
        self.assertEqual((suggestion.key, suggestion.preferred_direction), ("depth", "increase"))
        self.assertAlmostEqual(suggestion.mean_observed_delta, 0.3)


class IsolationAndCheckpointTests(unittest.TestCase):
    def test_docker_runner_applies_hardening_flags(self) -> None:
        class CaptureRunner:
            command = ()

            def run(self, command, *, cwd, env, timeout_seconds):
                self.command = tuple(command)
                return RunResult(self.command, 0, "", "", 0.01)

        with tempfile.TemporaryDirectory() as directory:
            runner = DockerRunner("agent@sha256:example")
            capture = CaptureRunner()
            runner.local = capture
            runner.run(
                ("python", "solve.py"), cwd=Path(directory),
                env={"ATLAS_SEED": "1"}, timeout_seconds=10,
            )
            command = capture.command
            self.assertIn("none", command)
            self.assertIn("no-new-privileges", command)
            self.assertIn("--read-only", command)
            self.assertIn("--cap-drop", command)

    def test_private_tests_are_revealed_only_after_solver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            private = root / "private"
            public.mkdir()
            private.mkdir()
            solver = root / "solver.py"
            solver.write_text(
                "from pathlib import Path\n"
                "assert not Path('.atlas-private').exists()\n"
                "Path('answer.txt').write_text('ok')\n",
                encoding="utf-8",
            )
            (private / "check.py").write_text(
                "from pathlib import Path\n"
                "assert Path('answer.txt').read_text() == 'ok'\n",
                encoding="utf-8",
            )
            task = TaskSpec(
                "hidden", "holdout", public,
                (sys.executable, ".atlas-private/check.py"), "hidden",
                private_evaluator_dir=private, seeds=(0,),
            )
            outcome = CommandTaskRunner(
                LocalProcessRunner(), (sys.executable, str(solver))
            ).evaluate(task, {}, 0)
            self.assertTrue(outcome.passed)
            (private / "review.py").write_text(
                "raise SystemExit(1)\n", encoding="utf-8"
            )
            reviewed_task = TaskSpec(
                "dual", "holdout", public,
                (sys.executable, ".atlas-private/check.py"), "dual judged",
                private_evaluator_dir=private,
                review_command=(sys.executable, ".atlas-private/review.py"),
                seeds=(0,),
            )
            reviewed = CommandTaskRunner(
                LocalProcessRunner(), (sys.executable, str(solver))
            ).evaluate(reviewed_task, {}, 0)
            self.assertFalse(reviewed.passed)
            self.assertIsNotNone(reviewed.reviewer)

    def test_local_runner_marks_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = LocalProcessRunner().run(
                (sys.executable, "-c", "import time; time.sleep(1)"),
                cwd=Path(directory), env={}, timeout_seconds=0.05,
            )
            self.assertTrue(result.timed_out)
            self.assertEqual(result.exit_code, 124)

    def test_checkpoint_gate_and_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"weights-v1")
            registry = CheckpointRegistry(root / "registry.json")
            registry.register(
                "v1", model, parent_id=None,
                metrics={"holdout": 0.9}, anchors_passed=True,
            )
            self.assertFalse(
                registry.promote(
                    "v1", constitution=Constitution(objective="locked"),
                    minimum_metric=0.8,
                )
            )
            self.assertTrue(
                registry.promote(
                    "v1",
                    constitution=Constitution(
                        objective="enabled", allow_weight_updates=True
                    ),
                    minimum_metric=0.8,
                    operator_approved=True,
                )
            )
            self.assertTrue(registry.verify("v1"))
            model.write_bytes(b"tampered")
            self.assertFalse(registry.verify("v1"))


class EndToEndTests(unittest.TestCase):
    def test_example_cli_and_resume(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config = json.loads(
            (repository / "examples" / "experiment.json").read_text(encoding="utf-8")
        )
        config["constitution"]["max_rounds"] = 1
        config["constitution"]["minimum_trials_for_ci"] = 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in config["tasks"]:
                task["seeds"] = [0]
                task["public_dir"] = str(repository / task["public_dir"])
                task["private_evaluator_dir"] = str(
                    repository / task["private_evaluator_dir"]
                )
            config["solver_command"] = [
                sys.executable, str(repository / "examples" / "toy_solver.py")
            ]
            config["state_path"] = str(root / "state.sqlite3")
            config["report_path"] = str(root / "report.md")
            config_path = root / "experiment.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(run_experiment(config_path), 0)
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("| 1 |", report)
            self.assertIn("r1-2-", report)

            with self.assertRaisesRegex(ValueError, "already contains"):
                run_experiment(config_path)

            config["resume"] = True
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(run_experiment(config_path), 0)
            resumed = (root / "report.md").read_text(encoding="utf-8")
            self.assertEqual(resumed.count("| 1 | `baseline` |"), 1)

            config["constitution"]["minimum_holdout_delta"] = 0.5
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run_experiment(config_path)


if __name__ == "__main__":
    unittest.main()
