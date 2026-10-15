"""独立补充实验入口：六进程checkpoint运行，可从相同协议继续。"""
from pathlib import Path
import argparse
import json
from paper1_active_req.experiment.reviewer_runner import run_supplement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    print(json.dumps(run_supplement(project / "research_artifacts/reviewer_supplement_20261009",project,args.workers)),flush=True)


if __name__ == "__main__":
    main()
