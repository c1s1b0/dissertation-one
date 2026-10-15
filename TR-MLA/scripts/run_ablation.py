from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.runner import run_constraint_state_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--training-specs", type=Path, required=True)
    parser.add_argument("--selected-weights", type=Path, required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument(
        "--ablated-term",
        choices=[
            "domain_reduction",
            "criticality",
            "conflict",
            "answer_cost",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema", type=Path, default=Path("configs/fields.yaml")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()
    written = run_constraint_state_ablation(
        tasks_path=args.tasks,
        training_specs_path=args.training_specs,
        schema_path=args.schema,
        config_path=args.config,
        selected_weights_path=args.selected_weights,
        variant_name=args.variant_name,
        ablated_term=args.ablated_term,
        output_path=args.output,
        seed=args.seed,
        budget=args.budget,
    )
    print(f"written_records={written}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
