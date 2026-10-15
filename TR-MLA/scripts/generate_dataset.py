from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.generator.complete_specs import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/fields.yaml"),
    )
    args = parser.parse_args()
    manifest = generate_dataset(
        seed=args.seed,
        count=args.count,
        output_dir=args.output,
        schema_path=args.schema,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
