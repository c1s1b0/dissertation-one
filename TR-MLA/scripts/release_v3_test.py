from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.generator.v3_relational_specs import release_v3_test_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="释放冻结后的 v3 test 集")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--confirm-release", action="store_true")
    parser.add_argument("--schema", type=Path, default=Path("configs/fields.yaml"))
    parser.add_argument(
        "--relation-config",
        type=Path,
        default=Path("configs/v3_relations.yaml"),
    )
    args = parser.parse_args(argv)
    if not args.confirm_release:
        raise ValueError("--confirm-release is required to release the v3 test set")
    manifest = release_v3_test_dataset(
        master_seed=20260806,
        output_dir=args.output,
        schema_path=args.schema,
        relation_config_path=args.relation_config,
        freeze_manifest_path=args.freeze_manifest,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
