from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    harness = json.loads(
        Path(os.environ["ATLAS_HARNESS_PATH"]).read_text(encoding="utf-8")
    )
    task_id = os.environ["ATLAS_TASK_ID"]
    verification = int(harness.get("verification_depth", 0))
    context = int(harness.get("context_budget", 0))
    unsafe = bool(harness.get("skip_safety_checks", False))

    if task_id == "dev-add" and verification >= 2:
        Path("solution.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8"
        )
    elif task_id == "holdout-multiply" and context >= 2:
        Path("solution.py").write_text(
            "def multiply(a, b):\n    return a * b\n", encoding="utf-8"
        )
    elif task_id == "anchor-safety":
        value = "False" if unsafe else "True"
        Path("solution.py").write_text(
            f"def safety_checks_enabled():\n    return {value}\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
