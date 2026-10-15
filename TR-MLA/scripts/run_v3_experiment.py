from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.v3_runner import (
    V3FreezeManifest,
    run_v3_selected_experiment,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行已释放且与冻结清单匹配的 v3 test 实验")
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args(argv)
    manifest = V3FreezeManifest.model_validate_json(
        args.freeze_manifest.read_text(encoding="utf-8")
    )
    dataset = args.dataset or Path(manifest.artifact_paths["development_manifest"]).parent
    written = run_v3_selected_experiment(
        test_tasks_path=dataset / "test_tasks.jsonl",
        training_specs_path=dataset / "train_complete_specs.jsonl",
        freeze_manifest_path=args.freeze_manifest,
        output_path=args.output,
    )
    print(written)


if __name__ == "__main__":
    main()
