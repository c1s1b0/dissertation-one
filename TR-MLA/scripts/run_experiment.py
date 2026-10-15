from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.runner import run_all_policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--training-specs", type=Path, required=True)
    parser.add_argument(
        "--schema", type=Path, default=Path("configs/fields.yaml")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--selected-weights", type=Path)
    args = parser.parse_args()
    written = run_all_policies(
        tasks_path=args.tasks,
        training_specs_path=args.training_specs,
        schema_path=args.schema,
        config_path=args.config,
        output_path=args.output,
        seed=args.seed,
        budget=args.budget,
        selected_weights_path=args.selected_weights,
    )
    print(f"written_records={written}")


if __name__ == "__main__":
    main()
