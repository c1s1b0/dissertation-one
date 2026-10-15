from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.generator.v2_pivotal_specs import generate_v2_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成论文一 v2 的独立枢纽字段数据集。"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--schema", type=Path, default=Path("configs/fields.yaml")
    )
    args = parser.parse_args()
    manifest = generate_v2_dataset(
        seed=args.seed, output_dir=args.output, schema_path=args.schema
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
