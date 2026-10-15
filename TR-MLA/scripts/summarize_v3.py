from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.v3_runner import summarize_v3_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="汇总完整的 1800 条 v3 formal 记录")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args(argv)
    summarize_v3_results(
        episodes_path=args.episodes,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
