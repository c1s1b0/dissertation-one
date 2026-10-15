"""独立固定序列噪声重放入口。"""
import argparse
import json
from pathlib import Path
from paper1_active_req.experiment.reviewer_noise_replay import run_noise_replay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='固定历史查询序列含噪输入保护测试')
    parser.add_argument('--output-dir', type=Path, default=Path('research_artifacts/reviewer_supplement_20261009/diagnostics/noise_replay'))
    args = parser.parse_args()
    summary = run_noise_replay(Path(__file__).resolve().parents[1], args.output_dir.resolve())
    print(json.dumps({key:summary[key] for key in ('original_episode_count','replay_count','runtime_seconds','clean_metric_mismatch_n')}, ensure_ascii=False))
