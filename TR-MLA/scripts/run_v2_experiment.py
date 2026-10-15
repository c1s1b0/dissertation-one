from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.v2_runner import run_v2_selected_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="以 validation 选择的 λ 运行论文一 v2 test。"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--training-specs", type=Path, required=True)
    parser.add_argument("--selected-lambdas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/fields.yaml"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()
    written = run_v2_selected_experiment(
        tasks_path=args.tasks,
        training_specs_path=args.training_specs,
        schema_path=args.schema,
        selected_lambdas_path=args.selected_lambdas,
        output_path=args.output,
        seed=args.seed,
        budget=args.budget,
    )
    print(f"written_records={written}")


if __name__ == "__main__":
    main()
