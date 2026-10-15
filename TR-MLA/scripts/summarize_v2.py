from __future__ import annotations

import argparse
from pathlib import Path

from paper1_active_req.experiment.v2_runner import summarize_v2_results


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 v2 正式 episode 结果。")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize_v2_results(args.episodes, args.output)


if __name__ == "__main__":
    main()
