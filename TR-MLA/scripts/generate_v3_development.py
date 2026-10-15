from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.generator.v3_relational_specs import (
    generate_v3_development_dataset,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成 v3 开发集")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/fields.yaml"))
    parser.add_argument(
        "--relation-config",
        type=Path,
        default=Path("configs/v3_relations.yaml"),
    )
    args = parser.parse_args(argv)
    manifest = generate_v3_development_dataset(
        master_seed=20260806,
        output_dir=args.output,
        schema_path=args.schema,
        relation_config_path=args.relation_config,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
