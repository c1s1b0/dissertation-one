from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.v3_runner import run_v3_validation_tuning


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="仅在 validation split 上调节 v3 TR-MLA lambda_cost")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/fields.yaml"))
    parser.add_argument("--relation-config", type=Path, default=Path("configs/v3_relations.yaml"))
    args = parser.parse_args(argv)
    selection = run_v3_validation_tuning(
        validation_tasks_path=args.dataset / "validation_tasks.jsonl",
        training_specs_path=args.dataset / "train_complete_specs.jsonl",
        schema_path=args.schema,
        relation_config_path=args.relation_config,
        output_dir=args.output,
        seed=20260806,
        budget=8,
    )
    print(selection["selected_lambda"])


if __name__ == "__main__":
    main()
