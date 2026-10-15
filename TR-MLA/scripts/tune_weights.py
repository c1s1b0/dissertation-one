from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.tuning import (
    run_successive_halving_weight_search,
    run_weight_search,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-tasks", type=Path, required=True)
    parser.add_argument("--training-specs", type=Path, required=True)
    parser.add_argument(
        "--schema", type=Path, default=Path("configs/fields.yaml")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument(
        "--method",
        choices=["grid", "successive-halving"],
        default="successive-halving",
    )
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--stage-one-spec-count", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=8)
    args = parser.parse_args()
    common = {
        "validation_tasks_path": args.validation_tasks,
        "training_specs_path": args.training_specs,
        "schema_path": args.schema,
        "config_path": args.config,
        "records_path": args.records,
        "selected_path": args.selected,
        "seed": args.seed,
        "budget": args.budget,
    }
    if args.method == "grid":
        best = run_weight_search(**common)
    else:
        best = run_successive_halving_weight_search(
            **common,
            candidate_count=args.candidate_count,
            stage_one_spec_count=args.stage_one_spec_count,
            finalists=args.finalists,
        )
    print(best.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
