"""运行独立审稿诊断，不修改旧实验输出。"""
from pathlib import Path
import argparse
import json
from paper1_active_req.experiment.reviewer_diagnostics import run_diagnostics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="审稿补充连续校准和约束压力诊断")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/generated/v3"))
    parser.add_argument("--label", default="seed20260806")
    parser.add_argument("--output-dir", type=Path, default=Path("research_artifacts/reviewer_supplement_20261009/diagnostics"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_diagnostics(root, args.dataset_dir.resolve(), args.output_dir.resolve(), args.label)
    print(json.dumps({"dataset": result["dataset"], "train_n": result["train_n"], "test_n": result["test_n"],
                      "continuous_rows": result["continuous"]["row_count"], "stress": result["stress"]}, ensure_ascii=False, indent=2))
