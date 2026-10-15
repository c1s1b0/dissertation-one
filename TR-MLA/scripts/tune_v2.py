from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.v2_runner import run_v2_validation_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="仅使用 validation split 调整 v2 λ。")
    parser.add_argument("--validation-tasks", type=Path, required=True)
    parser.add_argument("--training-specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/fields.yaml"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()
    selected = run_v2_validation_tuning(
        validation_tasks_path=args.validation_tasks,
        training_specs_path=args.training_specs,
        schema_path=args.schema,
        output_dir=args.output,
        seed=args.seed,
        budget=args.budget,
    )
    print(selected["selected_lambdas"])


if __name__ == "__main__":
    main()
