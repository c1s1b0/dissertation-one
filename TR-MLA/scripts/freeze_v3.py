from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.v3_runner import (
    formal_v3_source_paths,
    freeze_v3_artifacts,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="冻结 v3 validation 选择及可复核工件哈希")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/fields.yaml"))
    parser.add_argument("--relation-config", type=Path, default=Path("configs/v3_relations.yaml"))
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/v3_experiment.yaml"))
    args = parser.parse_args(argv)
    manifest = freeze_v3_artifacts(
        development_manifest_path=args.dataset / "development_manifest.json",
        selection_file=args.selection,
        schema_path=args.schema,
        relation_config_path=args.relation_config,
        experiment_config_path=args.experiment_config,
        source_paths=formal_v3_source_paths(),
        output_path=args.output,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
