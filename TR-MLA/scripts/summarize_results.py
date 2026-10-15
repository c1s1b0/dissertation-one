from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.runner import (
    summarize_ablation_results,
    summarize_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stratify-by-missing-rate", action="store_true"
    )
    parser.add_argument("--reference-episodes", type=Path)
    parser.add_argument(
        "--ablation-episodes", type=Path, nargs="+"
    )
    parser.add_argument("--ablation-output", type=Path)
    args = parser.parse_args()
    ablation_arguments = [
        args.reference_episodes,
        args.ablation_episodes,
        args.ablation_output,
    ]
    if any(value is not None for value in ablation_arguments) and not all(
        value is not None for value in ablation_arguments
    ):
        parser.error(
            "--reference-episodes, --ablation-episodes, and "
            "--ablation-output must be provided together"
        )
    summarize_results(
        args.episodes,
        args.output_dir,
        stratify_by_missing_rate=args.stratify_by_missing_rate,
    )
    if args.reference_episodes is not None:
        summarize_ablation_results(
            args.reference_episodes,
            args.ablation_episodes,
            args.ablation_output,
        )
    print(f"summary_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
